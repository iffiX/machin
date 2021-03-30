from typing import Callable, Any, Union, List, Tuple, Dict
from machin.parallel.distributed import get_world, get_cur_name
from machin.parallel.server import PushPullGradServerImpl, PushPullModelServerImpl
from torch.optim import Adam


def grad_server_helper(
    model_creators: List[Callable],
    group_name: str = "grad_server",
    members: Union[str, List[str]] = "all",
    optimizer: Any = Adam,
    learning_rate: Union[float, List[float]] = 1e-3,
    optimizer_kwargs: List[Dict[str, Any]] = None,
    lr_scheduler: Any = None,
    lr_scheduler_args: List[Tuple] = None,
    lr_scheduler_kwargs: List[Dict[str, Any]] = None,
):
    """
    Helper function for creating a tuple of grad servers,
    used by A3C, IMPALE, etc. This function requires all processes
    in the world to enter.

    Args:
        model_creators: A list of model creator functions,
            each one corresponds to one gradient reduction server.
        group_name: Name of the RPC group where gradient servers should be
            registered on, the group name should be unique.
        members: Name of the involved RPC processes, ``"all"`` for all
            processes, they will be used as secondary reducers, the first
            process will be the primary reducer.
        optimizer: Optimizer class, default is Adam.
        learning_rate: Learning rate of each optimizer. Or a single float value
            for every one.
        optimizer_kwargs: Optimizer keyword arguments for each optimizer of
            each model.
        lr_scheduler: Learning rate scheduler class.
        lr_scheduler_args: Learning rate scheduler arguments for each
            lr_scheduler corresponding to each optimizer.
        lr_scheduler_kwargs: Learning rate scheduler keyword arguments for each
            lr_scheduler corresponding to each optimizer.

    Returns:
        A tuple of accessors to gradient servers, the tuple has the
        same size as ``model_creators``
    """
    # Note:
    # passing a list of creator functions instead of passing a list of models
    # directly is designed to remove the unnecessary model creation cost on
    # not-the-primary-reducer processes.

    # create groups first
    world = get_world()
    members = world.get_members() if members == "all" else members
    server_group = world.create_rpc_group(group_name, members)

    if isinstance(learning_rate, float):
        learning_rate = [learning_rate] * len(model_creators)

    optimizer_kwargs = optimizer_kwargs or [{}] * len(model_creators)
    lr_scheduler_args = lr_scheduler_args or [()] * len(model_creators)
    lr_scheduler_kwargs = lr_scheduler_kwargs or [{}] * len(model_creators)

    # create servers
    primary_reducer = members[0]
    servers = [
        PushPullGradServerImpl(
            "grad_server_" + str(i), server_group, primary_reducer=primary_reducer
        )
        for i in range(len(model_creators))
    ]
    if get_cur_name() == primary_reducer:
        for (
            model_creator,
            server,
            optim_kwargs,
            lr,
            lr_sch_args,
            lr_sch_kwargs,
        ) in zip(
            model_creators,
            servers,
            optimizer_kwargs,
            learning_rate,
            lr_scheduler_args,
            lr_scheduler_kwargs,
        ):
            model = model_creator()
            if lr_scheduler is None:
                server.manage_model(
                    model, optimizer(model.parameters(), lr=lr, **optim_kwargs)
                )
            else:
                optimizer = optimizer(model.parameters(), lr=lr, **optim_kwargs)
                server.manage_model(
                    model,
                    optimizer,
                    lr_scheduler(optimizer, *lr_sch_args, **lr_sch_kwargs),
                )
            server.start()

    server_group.barrier()
    servers = tuple(
        server_group.get_paired("grad_server_" + str(i)).to_here()
        for i in range(len(model_creators))
    )

    # accessors instead of actual implementation instance
    # will be returned because of __reduce__
    return servers


def model_server_helper(
    model_num: int,
    group_name: str = "model_server",
    members: Union[str, List[str]] = "all",
):
    """
    Helper function for creating a tuple of model servers,
    used by APEX, etc. This function requires all processes
    in the world to enter.

    Args:
        model_num: The number of models, corresponds to the number of model
            servers, since each server manages 1 model.
        group_name: Name of the RPC group where gradient servers should be
            registered on, the group name should be unique.
        members: Name of the involved RPC processes, ``"all"`` for all
            processes, only the first process will serve as the server in the
            current implementation.

    Returns:
        A tuple of accessors to model servers, the size of tuple is
        ``model_num``
    """
    # create groups first
    world = get_world()
    members = world.get_members() if members == "all" else members
    server_group = world.create_rpc_group(group_name, members)

    # create servers
    # In current implementation, only one process will initialize the server
    if get_cur_name() == members[0]:
        for i in range(model_num):
            _server = PushPullModelServerImpl("model_server_" + str(i), server_group)

    server_group.barrier()

    servers = tuple(
        server_group.get_paired("model_server_" + str(i)).to_here()
        for i in range(model_num)
    )

    # accessors instead of actual implementation instance
    # will be returned because of __reduce__
    return servers
