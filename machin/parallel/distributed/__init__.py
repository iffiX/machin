import warnings

try:
    from ._world import (
        World,
        CollectiveGroup,
        RpcGroup,
        get_world,
        get_cur_rank,
        get_cur_name,
        is_world_initialized,
        debug_with_process,
    )

    from . import _world as world
except Exception as e:
    warnings.warn(
        f"""
        
        Importing world failed.
        Exception: {str(e)}
        This might be because you are using platforms other than linux.
        All exported symbols will be set to `None`, please don't use
        any distributed framework.
        """
    )
    World = None
    CollectiveGroup = None
    RpcGroup = None
    get_world = None
    get_cur_rank = None
    get_cur_name = None
    is_world_initialized = None
    debug_with_process = None

__all__ = [
    "World",
    "CollectiveGroup",
    "RpcGroup",
    "get_world",
    "get_cur_rank",
    "get_cur_name",
    "is_world_initialized",
    "debug_with_process",
]
