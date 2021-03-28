import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import DataLoader
from .config import init_algorithm_from_config
from .pl_plugin import DDPPlugin, DDPSpawnPlugin


class Launcher(pl.LightningModule):
    def __init__(self, config, env_dataset_creator):
        super(Launcher, self).__init__()
        self.config = config
        self.env_dataset_creator = env_dataset_creator
        self.frame = init_algorithm_from_config(config)
        self._frame_pl_inited = False

        # forward models to the launcher module, so that parameters are handled.
        for name, model in zip(self.frame.get_top_model_names(),
                               self.frame.top_models):
            self.add_module(name, model)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.env_dataset_creator(self.frame,
                                             self.config["train_env_config"]),
            collate_fn=lambda x: x
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.env_dataset_creator(self.frame,
                                             self.config["test_env_config"]),
            collate_fn=lambda x: x
        )

    def training_step(self, batch, _batch_idx):
        self._init_frame_with_pl()
        self.frame.store_episode(batch[0].observations)
        for _ in range(len(batch[0].observations)):
            self.frame.update()
        self._log(batch[0].logs)

    def test_step(self, batch, _batch_idx):
        self._init_frame_with_pl()
        self.frame.store_episode(batch[0].observations)
        for _ in range(len(batch[0].observations)):
            self.frame.update()
        self._log(batch[0].logs)

    def configure_optimizers(self):
        return self.frame.optimizers, self.frame.lr_schedulers

    def _init_frame_with_pl(self):
        if not self._frame_pl_inited:
            acc_con = self.trainer.accelerator_connector
            if (not self.frame.is_distributed() and
                    acc_con.use_ddp()):
                raise RuntimeError("Current framework: {} is not a distributed "
                                   "framework, you should not use an "
                                   "accelerator.".format(self.config["frame"]))

            if (self.frame.is_distributed() and
                    (acc_con.use_ddp() or
                     type(acc_con.training_type_plugin) not in
                     (DDPPlugin, DDPSpawnPlugin))):
                raise RuntimeError("Current framework: {} is a distributed "
                                   "framework, you should initialize the "
                                   "trainer with a ddp type accelerator, and "
                                   "pass in overloaded DDPPLugin or "
                                   "DDPSpawnPlugin from auto.pl_plugin."
                                   .format(self.config["frame"]))
            self.frame.set_backward_function(self.manual_backward)
            self.frame.optimizers = self.optimizers()
            self._frame_pl_inited = True

    def _log(self, logs):
        for log_key, log_val in logs:
            if isinstance(log_val, tuple) and callable(log_val[1]):
                log_val[1](self, log_key, log_val[0])
            else:
                is_dist_initialized = dist.is_available() and \
                                      dist.is_initialized()
                self.log(log_key, log_val, sync_dist=is_dist_initialized)
