from multiprocessing import get_context, get_start_method
from .pool import ThreadPool, Pool
from .distributed import Group, World, get_cur_name, get_cur_real_name
from .pickle_helper import mark_static_module

__all__ = [
    "get_context",
    "get_start_method",
    "get_cur_name",
    "get_cur_real_name",
    "mark_static_module",
    "ThreadPool", "Pool", "Group", "World",
]
