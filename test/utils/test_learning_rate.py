from machin.utils.logging import default_logger
from machin.utils.learning_rate import gen_learning_rate_func


def test_gen_learning_rate_func():
    """
    Generate learning learning learning function.

    Args:
    """
    func = gen_learning_rate_func([(0, 1e-3), (20000, 1e-3)], default_logger)
    func(10000)
    func(20001)
