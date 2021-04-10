from typing import Union, List, Dict, Any, Callable
from torch import distributed as dist
from torch.utils.data.dataloader import DataLoader
from .dataset import RLDataset
from .config import init_algorithm_from_config, is_algorithm_distributed
from .pl_plugin import DDPPlugin, DDPSpawnPlugin
from machin.parallel.distributed import get_world, debug_with_process as debug
from machin.frame.algorithms import TorchFramework
from machin.utils.conf import Config
import pytorch_lightning as pl


class Launcher(pl.LightningModule):
    def __init__(
        self,
        config: Union[Dict[str, Any], Config],
        env_dataset_creator: Callable[[TorchFramework, Dict[str, Any]], RLDataset],
    ):
        """
        Launcher module for all algorithm frameworks.

        Args:
            config: All configs, including frame config, train env config, etc.
            env_dataset_creator: A callable which accepts the algorithm frame
                and env config dictionary, and outputs a environment dataset.
        """
        super().__init__()
        self.config = config
        self.env_dataset_creator = env_dataset_creator
        assert not is_algorithm_distributed(config)
        self.frame = init_algorithm_from_config(self.config, model_device=self.device)
        self._frame_pl_inited = False
        self.automatic_optimization = False

        # forward models to the launcher module, so that parameters are handled.
        for name, model in zip(self.frame.get_top_model_names(), self.frame.top_models):
            self.add_module(name, model)

    def on_train_start(self):
        self._init_frame_with_pl()

    def on_test_start(self):
        self._init_frame_with_pl()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.env_dataset_creator(
                self.frame, self.config["train_env_config"]
            ),
            collate_fn=lambda x: x,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.env_dataset_creator(
                self.frame, self.config["test_env_config"]
            ),
            collate_fn=lambda x: x,
        )

    def training_step(self, batch, _batch_idx):
        for idx, sample in enumerate(batch):
            self.frame.store_episode(sample.observations)
            for _ in range(len(sample.observations)):
                self.frame.update()
            self._log(sample.logs)

    def test_step(self, batch, _batch_idx):
        for sample in batch:
            self._log(sample.logs)

    def configure_optimizers(self):
        return self.frame.optimizers, self.frame.lr_schedulers

    def _init_frame_with_pl(self):
        if not self._frame_pl_inited:
            acc_con = self.trainer.accelerator_connector

            if acc_con.use_ddp:
                raise RuntimeError(
                    f"Current framework: {self.config['frame']} is not a "
                    "distributed framework, you should not use an accelerator."
                )

            # Replace optimizer function with pytorch lightning
            # We can probably support DDP automatically for non-distributed
            # models in this way in the future
            self.frame.set_backward_function(self.manual_backward)
            self.frame.optimizers = (
                self.optimizers()
                if isinstance(self.optimizers(), list)
                else [self.optimizers()]
            )
            self._frame_pl_inited = True

    def _log(self, logs: List[Dict[str, Any]]):
        for log in logs:
            for log_key, log_val in log.items():
                if isinstance(log_val, tuple) and callable(log_val[1]):
                    log_val[1](self, log_key, log_val[0])
                else:
                    self.log(log_key, log_val, prog_bar=True)


class DistributedLauncher(pl.LightningModule):
    def __init__(
        self,
        config: Union[Dict[str, Any], Config],
        env_dataset_creator: Callable[[TorchFramework, Dict[str, Any]], RLDataset],
    ):
        """
        Launcher module for all algorithm frameworks.

        Args:
            config: All configs, including frame config, train env config, etc.
            env_dataset_creator: A callable which accepts the algorithm frame
                and env config dictionary, and outputs a environment dataset.
        """
        super().__init__()
        self.config = config
        self.env_dataset_creator = env_dataset_creator
        assert is_algorithm_distributed(config)
        # Lazy initialization for distributed frameworks, framework must be initialized
        # after initializing world.
        self.frame = None
        self.group = None
        # early stopping monitor
        self._train_esm = None
        self._test_esm = None
        self._frame_pl_inited = False
        self._batch_num = None
        self.automatic_optimization = False

    def on_train_start(self):
        self._init_frame_with_pl()

    def on_test_start(self):
        self._init_frame_with_pl()

    def train_dataloader(self):
        dataset = self.env_dataset_creator(self.frame, self.config["train_env_config"])
        self._train_esm = dataset.early_stopping_monitor
        return DataLoader(
            dataset=dataset, batch_size=self._batch_num, collate_fn=lambda x: x
        )

    def test_dataloader(self):
        dataset = self.env_dataset_creator(self.frame, self.config["test_env_config"])
        self._test_esm = dataset.early_stopping_monitor
        return DataLoader(dataset=dataset, collate_fn=lambda x: x)

    def training_step(self, batch, batch_idx):
        debug(f"Begin training step batch {batch_idx}")
        for idx, sample in enumerate(batch):
            self.frame.store_episode(sample.observations)
            debug("Begin update")
            if self.current_epoch != 0 or batch_idx != 0:
                # Skip the first batch so that buffers can be filled
                for _ in range(len(sample.observations)):
                    self.frame.update()
            debug("End update")
            if idx == 0:
                # only log the first batch result, since different
                # processes may have different number of batches.
                # Otherwise barrier in log will fail
                self._log(sample.logs)
            debug("Logging finished")

        # Make sure they exit together
        self.group.barrier()
        debug(f"End training step batch {batch_idx}")

    def test_step(self, batch, _batch_idx):
        for sample in batch:
            self._log(sample.logs)

    def configure_optimizers(self):
        # Distributed case
        # Sets ``lr_schedulers`` and ``optimizers`` attribute of trainer.accelerator
        # later when framework is initialized by overloaded DDP plugins.
        return None

    def init_frame(self):
        # Called by overloaded pytorch lightning DDP plugins
        if self.frame is None:
            # initialize framework
            self.frame = init_algorithm_from_config(
                self.config, model_device=self.device
            )
            self._batch_num = self.config["batch_num"].get(self.frame.role, 1)
            # forward models to the launcher module, so that parameters are handled.
            for name, model in zip(
                self.frame.get_top_model_names(), self.frame.top_models
            ):
                self.add_module(name, model)

            # create group for custom synchronization with large timeout
            # otherwise the default barrier in pytorch_lightning after training step
            # will throw an exception.
            world = get_world()
            self.group = world.create_collective_group(world.get_ranks(), timeout=86400)

    def _init_frame_with_pl(self):
        if not self._frame_pl_inited:
            acc_con = self.trainer.accelerator_connector

            # For distributed frame, use default settings, don't change optimizers.
            if not acc_con.use_ddp or type(acc_con.training_type_plugin) not in (
                DDPPlugin,
                DDPSpawnPlugin,
            ):
                raise RuntimeError(
                    f"Current framework: {self.config['frame']} is a distributed "
                    "framework, you should initialize the "
                    "trainer with a ddp type accelerator, and "
                    "must import machin.auto package to patch"
                    "the default DDP plugin."
                )

            self._frame_pl_inited = True

    def _log(self, logs: List[Dict[str, Any]]):
        for log in logs:
            for log_key, log_val in log.items():
                if isinstance(log_val, tuple) and callable(log_val[1]):
                    log_val[1](self, log_key, log_val[0])
                else:
                    is_dist_initialized = dist.is_available() and dist.is_initialized()
                    # only reduce the early-stopping monitor
                    # since other keys may not be presented in other processes
                    if log_key in (self._train_esm, self._test_esm):
                        self.log(
                            log_key,
                            log_val,
                            prog_bar=True,
                            sync_dist=is_dist_initialized,
                        )
                    else:
                        self.log(log_key, log_val, prog_bar=True)
