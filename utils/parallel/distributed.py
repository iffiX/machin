from datetime import timedelta
from torch.distributed import *
from torch.distributed import rpc


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(worker, method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(worker, call_method, args=args, kwargs=kwargs)


class World:
    def __init__(self,
                 world_size,
                 current_rank,
                 backend="gloo",
                 init_method="tcp://localhost:9100",
                 timeout=60,
                 ):
        self.world_size = world_size
        self.current_rank = current_rank

        rpc.init_rpc("{}".format(current_rank),
                     rank=current_rank,
                     world_size=world_size,
                     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                         init_method=init_method
                     ))
        self.groups = {}


    def create_group(self, group_name, ranks):
        rpc.
