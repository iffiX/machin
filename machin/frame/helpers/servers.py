from typing import Callable, Any, List
from machin.parallel.distributed import (
    get_world, get_cur_name
)
from machin.parallel.server import (
    PushPullGradServerImpl,
    PushPullModelServerImpl
)
from torch.optim import Adam


def grad_server_helper(model_creators: List[Callable],
                       optimizer: Any = Adam,
                       learning_rate: float = 1e-3):
    """
    Helper function for creating a tuple of grad servers,
    used by A3C, IMPALE, etc. This function requires all processes
    in the world to enter.

    Warning:
        You should never run this function twice!

    Args:
        model_creators: A list of model creator functions,
            each one corresponds to one gradient reduction server.
        optimizer: Optimizer type, default is Adam.
        learning_rate: Learning rate of the optimizer.

    Returns:
        A tuple of accessors to gradient servers, the tuple has the
        same size as ``model_creators``
    """
    # Note:
    # passing a list of creator functions instead of passing a list of models
    # directly is designed to remove the unnecessary model creation cost on
    # not-the-primary-reducer processes.
    DEFAULT_GROUP_NAME = "server_group"

    # create groups first
    world = get_world()
    server_group = world.create_rpc_group(DEFAULT_GROUP_NAME,
                                          world.get_members())

    # create servers
    primary_reducer = world.get_members()[0]
    servers = [
        PushPullGradServerImpl("grad_server_" + str(i),
                               server_group,
                               primary_reducer=primary_reducer)
        for i in range(len(model_creators))
    ]
    if get_cur_name() == primary_reducer:
        for model_creator, server in zip(model_creators, servers):
            model = model_creator()
            server.manage_model(model,
                                optimizer(model.parameters(),
                                          lr=learning_rate))
            server.start()

    server_group.barrier()
    servers = tuple(
        server_group.get_paired("grad_server_" + str(i)).to_here()
        for i in range(len(model_creators))
    )

    # accessors instead of actual implementation instance
    # will be returned because of __reduce__
    return servers


def model_server_helper(model_num):
    """
    Helper function for creating a tuple of model servers,
    used by APEX, etc. This function requires all processes
    in the world to enter.

    Warning:
        You should never run this function twice!

    Returns:
        A tuple of accessors to model servers, the size of tuple is
        ``model_num``
    """
    DEFAULT_GROUP_NAME = "server_group"

    # create groups first
    world = get_world()
    server_group = world.create_rpc_group(DEFAULT_GROUP_NAME,
                                          world.get_members())

    # create servers
    # In current implementation, only one process will initialize the server
    if get_cur_name() == world.get_members()[0]:
        for i in range(model_num):
            _server = PushPullModelServerImpl("model_server_" + str(i),
                                              server_group)

    server_group.barrier()

    servers = tuple(
        server_group.get_paired("model_server_" + str(i)).to_here()
        for i in range(model_num)
    )

    # accessors instead of actual implementation instance
    # will be returned because of __reduce__
    return servers
