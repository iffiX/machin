from typing import Callable, Any
from machin.parallel.distributed import (
    get_world, get_cur_name
)
from machin.parallel.server import (
    PushPullGradServerImpl,
    PushPullModelServerImpl
)
from time import sleep
from torch.optim import Adam


def grad_server_helper(actor_model_creator: Callable,
                       critic_model_creator: Callable,
                       optimizer: Any = Adam,
                       learning_rate: float = 1e-3):
    """
    Helper function for creating a pair of grad servers,
    used by A3C, IMPALE, etc. This function requires all processes
    in the world to enter.

    Warning:
        You should never run this function twice!

    Args:
        actor_model_creator: Function used to create the actor model.
        critic_model_creator: Function used to create the critic model.
        optimizer: Optimizer type, default is Adam.
        learning_rate: Learning rate of the optimizer.

    Returns:
        A tuple of two accessors to gradient servers,
        the first one is for actor, the second one is for critic.
    """
    DEFAULT_GROUP_NAME = "server_group"
    DEFAULT_SERVER_NAMES = ("grad_server_actor", "grad_server_critic")

    # create groups first
    world = get_world()
    server_group = world.create_rpc_group(DEFAULT_GROUP_NAME,
                                          world.get_members())

    # create servers
    primary_reducer = world.get_members()[0]
    actor_server = PushPullGradServerImpl(DEFAULT_SERVER_NAMES[0],
                                          server_group,
                                          primary_reducer=primary_reducer)
    critic_server = PushPullGradServerImpl(DEFAULT_SERVER_NAMES[1],
                                           server_group,
                                           primary_reducer=primary_reducer)
    if get_cur_name() == primary_reducer:
        actor_model = actor_model_creator()
        critic_model = critic_model_creator()
        actor_server.manage_model(actor_model,
                                  optimizer(actor_model.parameters(),
                                            lr=learning_rate))
        critic_server.manage_model(critic_model,
                                   optimizer(critic_model.parameters(),
                                             lr=learning_rate))
        actor_server.start()
        critic_server.start()
    server_group.barrier()
    actor_server = server_group.get_paired(DEFAULT_SERVER_NAMES[0]).to_here()
    critic_server = server_group.get_paired(DEFAULT_SERVER_NAMES[1]).to_here()

    # accessors instead of actual implementation instance
    # will be returned because of __reduce__
    return actor_server, critic_server


def model_server_helper():
    """
    Helper function for creating a pair of model servers,
    used by APEX, etc. This function requires all processes
    in the world to enter.

    Warning:
        You should never run this function twice!

    Returns:
        A tuple of two accessors to gradient servers,
        the first one is for actor, the second one is for critic.
    """
    DEFAULT_GROUP_NAME = "server_group"
    DEFAULT_SERVER_NAMES = ("model_server_actor", "model_server_critic")

    # create groups first
    world = get_world()
    server_group = world.create_rpc_group(DEFAULT_GROUP_NAME,
                                          world.get_members())

    # create servers
    # In current implementation, only one process will initialize the server

    if get_cur_name() == world.get_members()[0]:
        _actor_server = PushPullModelServerImpl(DEFAULT_SERVER_NAMES[0],
                                                server_group)
        _critic_server = PushPullModelServerImpl(DEFAULT_SERVER_NAMES[1],
                                                 server_group)
    server_group.barrier()

    actor_server = server_group.get_paired(DEFAULT_SERVER_NAMES[0]).to_here()
    critic_server = server_group.get_paired(DEFAULT_SERVER_NAMES[1]).to_here()

    # accessors instead of actual implementation instance
    # will be returned because of __reduce__
    return actor_server, critic_server
