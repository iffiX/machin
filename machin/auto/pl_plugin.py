import os
import logging
import pytorch_lightning as pl
from time import sleep
from typing import Optional
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin as DDP
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin as DDPS
from machin.parallel.distributed import World, is_world_initialized
from machin.parallel.process import Process
import multiprocessing as mp
import traceback

pl_logger = logging.getLogger("lightning")


def assert_world_config(cluster_env, master_addr, master_port, world_size):
    assert master_addr is not None, (
        "Master address not set, please set environment "
        "variable MASTER_ADDR or refer to the cluster environment "
        f"instructions you are using, current cluster env is {cluster_env}."
    )
    assert master_port is not None, (
        "Master port not set, please set environment "
        "variable MASTER_PORT or refer to the cluster environment "
        f"instructions you are using, current cluster env is {cluster_env}."
    )
    assert world_size is not None, (
        "World size not set, please set environment "
        "variable WORLD_SIZE or refer to the cluster environment "
        f"instructions you are using, current cluster env is {cluster_env}."
    )


class DDPPlugin(DDP):
    def init_ddp_connection(
        self, global_rank: Optional[int] = None, world_size: Optional[int] = None
    ) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()
        global_rank = (
            global_rank
            if global_rank is not None
            else self.cluster_environment.global_rank()
        )
        world_size = (
            world_size
            if world_size is not None
            else self.cluster_environment.world_size()
        )
        assert_world_config(
            self.cluster_environment, master_addr, master_port, world_size
        )

        if not is_world_initialized():
            pl_logger.info(
                f"initializing world: GLOBAL_RANK: {global_rank}, "
                f"MEMBER: {int(global_rank) + 1}/{world_size}"
            )
            # TODO: currently nccl is having problems with supporting
            # different configurations, use gloo as replacement.
            # See: https://github.com/pytorch/pytorch/issues/47885
            _w = World(
                name=str(global_rank),
                rank=int(global_rank),
                world_size=int(world_size),
                dist_backend="gloo",
                dist_init_method=f"tcp://{master_addr}:{master_port}",
                rpc_init_method=f"tcp://{master_addr}:{int(master_port) + 1}",
            )

    def configure_ddp(self):
        # initialize framework in the launcher
        self._model.init_frame()
        if self._model.frame.optimizers is not None:
            self._model.trainer.accelerator.optimizers = self._model.frame.optimizers
        if self._model.frame.lr_schedulers is not None:
            self._model.trainer.accelerator.lr_schedulers = (
                self._model.frame.lr_schedulers
            )
        super().configure_ddp()

    def training_step(self, *args, **kwargs):
        return self.lightning_module.training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.lightning_module.test_step(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.lightning_module.predict(*args, **kwargs)

    def post_training_step(self):
        pass


class DDPSpawnPlugin(DDPS):
    def init_ddp_connection(
        self, global_rank: Optional[int] = None, world_size: Optional[int] = None
    ) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()
        global_rank = (
            global_rank
            if global_rank is not None
            else self.cluster_environment.global_rank()
        )
        world_size = (
            world_size
            if world_size is not None
            else self.cluster_environment.world_size()
        )
        assert_world_config(
            self.cluster_environment, master_addr, master_port, world_size
        )

        if not is_world_initialized():
            pl_logger.info(
                f"initializing world: GLOBAL_RANK: {global_rank}, "
                f"MEMBER: {int(global_rank) + 1}/{world_size}"
            )
            # TODO: currently nccl is having problems with supporting
            # different configurations, use gloo as replacement.
            # See: https://github.com/pytorch/pytorch/issues/47885
            _w = World(
                name=str(global_rank),
                rank=int(global_rank),
                world_size=int(world_size),
                dist_backend="gloo",
                dist_init_method=f"tcp://{master_addr}:{master_port}",
                rpc_init_method=f"tcp://{master_addr}:{int(master_port) + 1}",
            )

    def configure_ddp(self):
        # initialize framework in the launcher
        self._model.init_frame()
        if self._model.frame.optimizers is not None:
            self._model.trainer.accelerator.optimizers = self._model.frame.optimizers
        if self._model.frame.lr_schedulers is not None:
            self._model.trainer.accelerator.lr_schedulers = (
                self._model.frame.lr_schedulers
            )
        super().configure_ddp()

    def start_training(self, trainer):
        self._spawn()
        trainer.optimizers = []

    def start_testing(self, trainer):
        self._spawn()

    def start_predicting(self, trainer):
        self._spawn()

    def _spawn(self):
        ctx = mp.get_context("spawn")
        processes = [
            Process(
                target=self.new_process,
                ctx=ctx,
                args=(i, self.lightning_module.trainer, self.mp_queue),
            )
            for i in range(self.mp_spawn_kwargs["nprocs"])
        ]
        for p in processes:
            p.start()
        while True:
            should_exit = False
            for p in processes:
                try:
                    p.watch()
                except Exception:
                    traceback.print_exc()
                    should_exit = True
            if should_exit:
                for p in processes:
                    p.terminate()
                    p.join()
                raise RuntimeError("One or more exceptions raised in sub-processes.")
            elif not all([p.is_alive() for p in processes]):
                break
            sleep(0.1)
        for p in processes:
            p.join()

    def training_step(self, *args, **kwargs):
        return self.lightning_module.training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.lightning_module.test_step(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.lightning_module.predict(*args, **kwargs)

    def post_training_step(self):
        pass


# monkey patching, since several initialization arguments of these two plugins
# are created by Trainer->Accelerator connectors->..., and cannot be retrieved
# before the trainer is initialized.
pl.trainer.connectors.accelerator_connector.DDPPlugin = DDPPlugin
pl.trainer.connectors.accelerator_connector.DDPSpawnPlugin = DDPSpawnPlugin
