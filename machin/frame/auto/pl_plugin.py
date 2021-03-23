import logging
from pytorch_lightning.plugins.training_type.ddp \
    import DDPPlugin as DDP
from pytorch_lightning.plugins.training_type.ddp_spawn \
    import DDPSpawnPlugin as DDPS
from machin.parallel.distributed import World, is_world_initialized


pl_logger = logging.getLogger("lightning")


class DDPPlugin(DDP):
    def configure_ddp(self):
        # Just remove wrapping model with DistributedDataParallel
        # in the original implementation.
        self.pre_configure_ddp()

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()

        if not is_world_initialized():
            pl_logger.info(
                "initializing world: GLOBAL_RANK: {}, MEMBER: {}/{}"
                .format(global_rank, global_rank + 1, world_size)
            )
            _w = World(name=str(global_rank),
                       rank=global_rank,
                       world_size=world_size,
                       dist_backend=self.torch_distributed_backend,
                       dist_init_method="tcp://{}:{}".format(master_addr,
                                                             master_port),
                       rpc_init_method="tcp://{}:{}".format(master_addr,
                                                            master_port + 1))


class DDPSpawnPlugin(DDPS):
    def configure_ddp(self):
        # Just remove wrapping model with DistributedDataParallel
        # in the original implementation.
        self.pre_configure_ddp()

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        master_addr = self.cluster_environment.master_address()
        master_port = self.cluster_environment.master_port()
        world_size = self.cluster_environment.world_size()

        if not is_world_initialized():
            pl_logger.info(
                "initializing world: GLOBAL_RANK: {}, MEMBER: {}/{}"
                    .format(global_rank, global_rank + 1, world_size)
            )
            _w = World(name=str(global_rank),
                       rank=global_rank,
                       world_size=world_size,
                       dist_backend=self.torch_distributed_backend,
                       dist_init_method="tcp://{}:{}".format(master_addr,
                                                             master_port),
                       rpc_init_method="tcp://{}:{}".format(master_addr,
                                                            master_port + 1))
