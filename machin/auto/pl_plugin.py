import logging
import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin as DDP
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin as DDPS
from machin.parallel.distributed import World, is_world_initialized


pl_logger = logging.getLogger("lightning")


def assert_world_config(cluster_env, master_addr, master_port, world_size):
    assert master_addr is not None, (
        "Master address not set, please set environment "
        "variable MASTER_ADDR or refer to the cluster environment "
        "instructions you are using, current cluster env is {}.".format(cluster_env)
    )
    assert master_port is not None, (
        "Master port not set, please set environment "
        "variable MASTER_PORT or refer to the cluster environment "
        "instructions you are using, current cluster env is {}.".format(cluster_env)
    )
    assert world_size is not None, (
        "World size not set, please set environment "
        "variable WORLD_SIZE or refer to the cluster environment "
        "instructions you are using, current cluster env is {}.".format(cluster_env)
    )


class DDPPlugin(DDP):
    def configure_ddp(self):
        # Just remove wrapping model with DistributedDataParallel
        # in the original implementation.
        self.pre_configure_ddp()

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()
        assert_world_config(
            self.cluster_environment, master_addr, master_port, world_size
        )

        if not is_world_initialized():
            pl_logger.info(
                "initializing world: GLOBAL_RANK: {}, MEMBER: {}/{}".format(
                    global_rank, int(global_rank) + 1, world_size
                )
            )
            _w = World(
                name=str(global_rank),
                rank=int(global_rank),
                world_size=int(world_size),
                dist_backend=self.torch_distributed_backend,
                dist_init_method=f"tcp://{master_addr}:{master_port}",
                rpc_init_method=f"tcp://{master_addr}:{int(master_port) + 1}",
            )

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
    def configure_ddp(self):
        # Just remove wrapping model with DistributedDataParallel
        # in the original implementation.
        self.pre_configure_ddp()

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()
        assert_world_config(
            self.cluster_environment, master_addr, master_port, world_size
        )

        if not is_world_initialized():
            pl_logger.info(
                "initializing world: GLOBAL_RANK: {}, MEMBER: {}/{}".format(
                    global_rank, int(global_rank) + 1, world_size
                )
            )
            _w = World(
                name=str(global_rank),
                rank=int(global_rank),
                world_size=int(world_size),
                dist_backend=self.torch_distributed_backend,
                dist_init_method=f"tcp://{master_addr}:{master_port}",
                rpc_init_method=f"tcp://{master_addr}:{int(master_port) + 1}",
            )

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
pl_logger.info("DDP plugin patched.")
