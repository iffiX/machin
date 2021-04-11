import os
import logging
import pytorch_lightning as pl
from time import sleep
from torch import distributed
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_only
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
    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()
        assert_world_config(
            self.cluster_environment, master_addr, master_port, world_size
        )

        if not is_world_initialized():
            pl_logger.info(
                f"initializing world: GLOBAL_RANK: {global_rank}, "
                f"MEMBER: {int(global_rank) + 1}/{world_size}"
            )
            # TODO: currently nccl is having problems with supporting
            # different cnfigurations, use gloo as replacement.
            # See: https://github.com/pytorch/pytorch/issues/47885
            _w = World(
                name=str(global_rank),
                rank=int(global_rank),
                world_size=int(world_size),
                dist_backend="gloo",
                dist_init_method=f"tcp://{master_addr}:{master_port}",
                rpc_init_method=f"tcp://{master_addr}:{int(master_port) + 1}",
            )

    def pre_dispatch(self):
        """
        This function is called by the trainer, before dispatch(), which
        starts training/testing/... etc.

        DDP plugin will initialize ddp connection at here.
        """

        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and distributed.is_initialized():
            pl_logger.info("-" * 100)
            pl_logger.info(f"distributed_backend={self.distributed_backend}")
            pl_logger.info(
                f"All DDP processes registered. Starting ddp with {self.world_size} "
                "processes"
            )
            pl_logger.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        # Move the launcher to the correct device
        # So in launcher, self.device would be set.
        self.model_to_device()

        # initialize framework in the launcher
        self._model.init_frame()
        if self._model.frame.optimizers is not None:
            self._model.trainer.accelerator.optimizers = self._model.frame.optimizers
        if self._model.frame.lr_schedulers is not None:
            self._model.trainer.accelerator.lr_schedulers = (
                self._model.frame.lr_schedulers
            )

        self.barrier()

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
    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()
        assert_world_config(
            self.cluster_environment, master_addr, master_port, world_size
        )

        if not is_world_initialized():
            pl_logger.info(
                f"initializing world: GLOBAL_RANK: {global_rank}, "
                f"MEMBER: {int(global_rank) + 1}/{world_size}"
            )
            # TODO: currently nccl is having problems with supporting
            # different cnfigurations, use gloo as replacement.
            # See: https://github.com/pytorch/pytorch/issues/47885
            _w = World(
                name=str(global_rank),
                rank=int(global_rank),
                world_size=int(world_size),
                dist_backend="gloo",
                dist_init_method=f"tcp://{master_addr}:{master_port}",
                rpc_init_method=f"tcp://{master_addr}:{int(master_port) + 1}",
            )

    def new_process(self, process_idx, trainer, mp_queue):
        """
        DDPSpawn does not pre-create processes. Instead it spawns processes as needed
        when training, testing, etc. is started, therefore we need to overload the
        process main function.
        """
        global trace_enabled
        self.mp_queue = mp_queue

        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and distributed.is_initialized():
            pl_logger.info("-" * 100)
            pl_logger.info(f"distributed_backend={self.distributed_backend}")
            pl_logger.info(
                f"All DDP processes registered. Starting ddp with {self.world_size} "
                "processes"
            )
            pl_logger.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        # Move the launcher to the correct device
        # So in launcher, self.device would be set.
        self.model_to_device()

        # initialize framework in the launcher
        self._model.init_frame()
        if self._model.frame.optimizers is not None:
            self._model.trainer.accelerator.optimizers = self._model.frame.optimizers
        if self._model.frame.lr_schedulers is not None:
            self._model.trainer.accelerator.lr_schedulers = (
                self._model.frame.lr_schedulers
            )

        self.barrier()

        results = trainer.train_or_test_or_predict()

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(results)

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
