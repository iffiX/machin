from machin.auto.config import generate_algorithm_config, generate_training_config
from machin.auto.envs.openai_gym import generate_env_config, launch
from machin.parallel.distributed import get_cur_rank, is_world_initialized
from machin.utils.logging import default_logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import ReduceOp
import os
import pickle
import torch as t
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


class LoggerDebugCallback(Callback):
    def on_train_start(self, *_, **__):
        from logging import DEBUG

        default_logger.setLevel(DEBUG)


class DDPInspectCallback(Callback):
    def __init__(self):
        self.max_total_reward = 0
        self.avg_max_total_reward = 0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, _batch_idx, _dataloader_idx
    ) -> None:
        for log in batch[0].logs:
            if "total_reward" in log:
                self.max_total_reward = max(log["total_reward"], self.max_total_reward)
                default_logger.info(
                    f"Process [{get_cur_rank()}] "
                    f"Current max total reward={self.max_total_reward:.2f}."
                )
                t_plugin = trainer.training_type_plugin
                self.avg_max_total_reward = self.reduce_max_total_reward(
                    trainer, t_plugin
                )
                trainer.should_stop = self.reduce_early_stopping_decision(
                    trainer, t_plugin
                )
                if trainer.should_stop:
                    default_logger.info(f"Process [{get_cur_rank()}] decides to exit.")
                return
        default_logger.error("Missing total reward in logs.")

    def reduce_early_stopping_decision(self, trainer, t_plugin):
        should_stop = t.tensor(
            int(self.max_total_reward >= 150), device=trainer.lightning_module.device
        )
        should_stop = t_plugin.reduce(should_stop, reduce_op=ReduceOp.SUM)
        should_stop = bool(should_stop == trainer.world_size)
        return should_stop

    def reduce_max_total_reward(self, trainer, t_plugin):
        avg = t.tensor(self.max_total_reward, device=trainer.lightning_module.device)
        avg = t_plugin.reduce(avg, reduce_op=ReduceOp.SUM)
        return float(avg)


if __name__ == "__main__":
    os.environ["WORLD_SIZE"] = "3"
    print(os.environ["TEST_SAVE_PATH"])
    config = generate_env_config("CartPole-v0", {})
    config = generate_training_config(root_dir=os.environ["ROOT_DIR"], config=config)
    config = generate_algorithm_config("DQNApex", config)

    # use ddp gpu
    config["gpus"] = [0, 0, 0]
    config["num_processes"] = 3
    # this testing process corresponds to this node
    config["num_nodes"] = 1
    config["early_stopping_patience"] = 100
    # Use class instead of string name since algorithms is distributed.
    config["frame_config"]["models"] = [QNet, QNet]
    config["frame_config"]["model_kwargs"] = [
        {"state_dim": 4, "action_num": 2},
        {"state_dim": 4, "action_num": 2},
    ]

    # cb = [DDPInspectCallback(), LoggerDebugCallback()]
    cb = [DDPInspectCallback()]
    launch(config, pl_callbacks=cb)
    if is_world_initialized() and get_cur_rank() == 0:
        with open(os.environ["TEST_SAVE_PATH"], "wb") as f:
            pickle.dump(cb[0].avg_max_total_reward, f)
