from machin.auto.config import (
    generate_algorithm_config,
    generate_env_config,
    generate_training_config,
    launch,
)
from pytorch_lightning.callbacks import Callback

import nni
import torch as t
import torch.nn as nn


class SomeQNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


class InspectCallback(Callback):
    def __init__(self):
        self.total_reward = 0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, _batch_idx, _dataloader_idx
    ) -> None:
        for l in batch[0].logs:
            if "total_reward" in l:
                self.total_reward = l["total_reward"]


if __name__ == "__main__":
    param = nni.get_next_parameter()
    cb = InspectCallback()
    while param:
        config = generate_algorithm_config("DQN")
        config = generate_env_config("openai_gym", config)
        config = generate_training_config(
            root_dir="trial", episode_per_epoch=10, max_episodes=10000, config=config
        )
        config["frame_config"]["models"] = ["SomeQNet", "SomeQNet"]
        config["frame_config"]["model_kwargs"] = [{"state_dim": 4, "action_num": 2}] * 2
        config["frame_config"]["learning_rate"] = param["lr"]
        config["frame_config"]["update_rate"] = param["upd"]
        launch(config, pl_callbacks=[cb])
        # we use total reward as "accuracy"
        nni.report_final_result(cb.total_reward)
        param = nni.get_next_parameter()
