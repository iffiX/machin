"""
This module is the place for all learning rate functions, currently, only
manual learning rate changing according to global steps is implemented,.
"""
from typing import List, Tuple
from logging import Logger


def gen_learning_rate_func(lr_map: List[Tuple[int, float]], logger: Logger = None):
    """
    Example::

        from torch.optim.lr_scheduler import LambdaLR

        # 0 <= step < 200000, lr=1e-3, 200000 <= step, lr=3e-4
        lr_func = gen_learning_rate_func([(0, 1e-3), (200000, 3e-4)],)
        lr_sch = LambdaLR(optimizer, lr_func)

    Args:
        lr_map: A 2d learning rate map, the first element of each row is step.
            the second is learning rate.
        logger: A logger to log current learning rate

    Returns:
        A learning rate generation function with signature `lr_gen(step)->lr`,
        accepts int and returns float. use it in your pytorch lr scheduler.
    """

    def learning_rate_func(step):
        for i in range(len(lr_map) - 1):
            if lr_map[i][0] <= step < lr_map[i + 1][0]:
                if logger is not None:
                    logger.info(f"Current learning rate:{lr_map[i][1]}")
                return lr_map[i][1]
        if logger is not None:
            logger.info(f"Current learning rate:{lr_map[-1][1]}")
        return lr_map[-1][1]

    return learning_rate_func
