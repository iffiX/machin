from logging import DEBUG
from machin.parallel.pool import Pool, P2PPool, CtxPool, ThreadPool, CtxThreadPool
from machin.utils.logging import default_logger as logger
from test.util_fixtures import *
from test.util_platforms import linux_only

import dill
import pytest
import torch as t


# enable pool logging
logger.setLevel(DEBUG)


def init_func(*_):
    print("Hello")


def func(x):
    return t.sum(x * 2)


def func2(x, y):
    return t.sum(x + y)


class TestPool:
    pool_impl = Pool

    def test_apply(self):
        pool = self.pool_impl(processes=2)
        x = t.ones([10])
        assert pool.apply(func, (x,)) == 20

        # for pytest-cov to run on sub processes
        pool.close()
        pool.join()

    def test_apply_async(self):
        pool = self.pool_impl(processes=2)
        x = t.ones([10])
        assert pool.apply_async(func, (x,)).get() == 20
        pool.close()
        pool.join()

    def test_map(self):
        pool = self.pool_impl(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(
            out == expect_out
            for out, expect_out in zip(pool.map(func, x), [0, 20, 40, 60, 80])
        )
        pool.close()
        pool.join()

    def test_map_async(self):
        pool = self.pool_impl(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(
            out == expect_out
            for out, expect_out in zip(
                pool.map_async(func, x).get(), [0, 20, 40, 60, 80]
            )
        )
        pool.close()
        pool.join()

    def test_imap(self):
        pool = self.pool_impl(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(
            out == expect_out
            for out, expect_out in zip(pool.imap(func, x), [0, 20, 40, 60, 80])
        )
        pool.close()
        pool.join()

    def test_imap_unordered(self):
        pool = self.pool_impl(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(
            out == expect_out
            for out, expect_out in zip(
                sorted(pool.imap_unordered(func, x)), [0, 20, 40, 60, 80]
            )
        )
        pool.close()
        pool.join()

    def test_starmap(self):
        pool = self.pool_impl(processes=2)
        x = [(t.ones([10]) * i, t.ones([10]) * i) for i in range(5)]
        assert all(
            out == expect_out
            for out, expect_out in zip(pool.starmap(func2, x), [0, 20, 40, 60, 80])
        )
        pool.close()
        pool.join()

    def test_starmap_async(self):
        pool = self.pool_impl(processes=2)
        x = [(t.ones([10]) * i, t.ones([10]) * i) for i in range(5)]
        assert all(
            out == expect_out
            for out, expect_out in zip(
                pool.starmap_async(func2, x).get(), [0, 20, 40, 60, 80]
            )
        )
        pool.close()
        pool.join()

    # Disabled for now
    # Individual testing passes while testing with all other module fails with:
    # Traceback (most recent call last):
    #   File "/opt/conda/lib/python3.7/multiprocessing/process.py", line 297,
    #   in _bootstrap
    #     self.run()
    #   File "/opt/conda/lib/python3.7/multiprocessing/process.py", line 99, in run
    #     self._target(*self._args, **self._kwargs)
    #   File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/pool.py",
    #   line 9, in clean_worker
    #     multiprocessing.pool.worker(*args, **kwargs)
    #   File "/opt/conda/lib/python3.7/multiprocessing/pool.py", line 110, in worker
    #     task = get()
    #   File "/var/lib/jenkins/workspace/machin_master_2/machin/parallel/queue.py",
    #   line 112, in get
    #     return loads(res)
    #   File "/opt/conda/lib/python3.7/site-packages/dill/_dill.py", line 283, in loads
    #     return load(file, ignore, **kwds)
    #   File "/opt/conda/lib/python3.7/site-packages/dill/_dill.py", line 278, in load
    #     return Unpickler(file, ignore=ignore, **kwds).load()
    #   File "/opt/conda/lib/python3.7/site-packages/dill/_dill.py", line 481, in load
    #     obj = StockUnpickler.load(self)
    #   File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing
    #   /reductions.py", line 117, in rebuild_cuda_tensor
    #     event_sync_required)
    # RuntimeError: CUDA error: peer access is not supported between these two devices

    # def test_gpu_tensor(self, gpu):
    #     x = [
    #         t.ones([10], device=gpu) * i
    #         for i in range(5)
    #     ]
    #     logger.info("GPU tensors created.")
    #     pool = self.pool_impl(processes=2, is_copy_tensor=True)
    #     logger.info("Pool 1 created.")
    #     assert all(
    #         out == expect_out
    #         for out, expect_out in zip(pool.map(func, x), [0, 20, 40, 60, 80])
    #     )
    #     pool.close()
    #     pool.join()
    #     logger.info("Pool 1 joined.")
    #
    #     pool = self.pool_impl(processes=2, is_copy_tensor=False, share_method="cuda")
    #     logger.info("Pool 2 created.")
    #     assert all(
    #         out == expect_out
    #         for out, expect_out in zip(pool.map(func, x), [0, 20, 40, 60, 80])
    #     )
    #     pool.close()
    #     pool.join()
    #     logger.info("Pool 2 joined.")

    @linux_only
    def test_cpu_shared_tensor(self):
        x = [t.ones([10]) * i for i in range(5)]
        for xx in x:
            xx.share_memory_()
        logger.info("CPU tensors created.")
        pool = self.pool_impl(processes=2, is_copy_tensor=False, share_method="cpu")
        logger.info("Pool created.")
        assert all(
            out == expect_out
            for out, expect_out in zip(pool.map(func, x), [0, 20, 40, 60, 80])
        )
        pool.close()
        pool.join()
        logger.info("Pool joined.")

    def test_lambda_and_local(self):
        x = [t.ones([10]) * i for i in range(5)]
        y = t.ones([10])

        x2 = [(t.ones([10]) * i, t.ones([10]) * i) for i in range(5)]

        def local_func(xx):
            nonlocal y
            return t.sum(xx + y)

        pool = self.pool_impl(processes=2, is_recursive=True)
        assert all(
            out == expect_out
            for out, expect_out in zip(pool.map(local_func, x), [10, 20, 30, 40, 50])
        )
        assert all(
            out == expect_out
            for out, expect_out in zip(
                pool.map(lambda xx: t.sum(xx[0] + xx[1]), x2), [0, 20, 40, 60, 80]
            )
        )
        pool.close()
        pool.join()

        pool = self.pool_impl(processes=2)
        assert all(
            out == expect_out
            for out, expect_out in zip(pool.map(func, x), [0, 20, 40, 60, 80])
        )
        pool.close()
        pool.join()

    def test_size(self):
        pool = self.pool_impl(processes=2)
        assert pool.size() == 2
        pool.close()
        pool.join()

    def test_reduce(self):
        with pytest.raises(NotImplementedError, match="cannot be passed"):
            dill.dumps(Pool(processes=2))


class TestP2PPool(TestPool):
    pool_impl = P2PPool


def ctx_func(ctx, x):
    # pretend to have done something using x wih context ctx
    return ctx, x


def ctx_func2(ctx, x, y):
    # pretend to have done something using x and y wwih context ctx
    return ctx, x + y


class TestCtxPool:
    def test_init(self):
        with pytest.raises(ValueError, match="not equal to the number"):
            _ = CtxPool(processes=2, worker_contexts=[0, 1, 2, 3])
        pool = CtxPool(processes=2, initializer=init_func, initargs=("some_args",))
        pool.close()
        pool.join()
        pool = CtxPool(processes=2, worker_contexts=[0, 1])
        pool.close()
        pool.join()

    def test_ctx_map(self):
        pool = CtxPool(processes=2, worker_contexts=[0, 1])
        # make sure both workers will have items
        x = [i for i in range(10)]
        result = pool.map(ctx_func, x)
        assert sorted([r[1] for r in result]) == x
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_ctx_starmap(self):
        pool = CtxPool(processes=2, worker_contexts=[0, 1])
        # make sure both workers will have items
        xy = [(i, i) for i in range(10)]
        result = pool.starmap(ctx_func2, xy)
        assert sorted([r[1] for r in result]) == [i * 2 for i in range(10)]
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_multiple_ctx_map(self):
        # test whether two context pools will interfere with each other
        pool = CtxPool(processes=2, worker_contexts=[0, 1])
        pool2 = CtxPool(processes=2, worker_contexts=[2, 3])
        # create enough work items so that the execution period of two pools
        # will overlap
        x = [i for i in range(50000)]
        result = pool.map(ctx_func, x)
        result2 = pool2.map(ctx_func, x)
        assert sorted([r[1] for r in result]) == x
        assert {r[0] for r in result}.issubset({0, 1})
        assert sorted([r[1] for r in result2]) == x
        assert {r[0] for r in result2}.issubset({2, 3})
        pool.close()
        pool.join()
        pool2.close()
        pool2.join()


class TestThreadPool:
    def test_size(self):
        pool = ThreadPool(processes=2)
        assert pool.size() == 2
        pool.close()
        pool.join()

    def test_reduce(self):
        with pytest.raises(NotImplementedError, match="cannot be passed"):
            dill.dumps(ThreadPool(processes=2))


class TestCtxThreadPool:
    def test_init(self):
        with pytest.raises(ValueError, match="not equal to the number"):
            _ = CtxThreadPool(processes=2, worker_contexts=[0, 1, 2, 3])
        pool = CtxThreadPool(
            processes=2, initializer=init_func, initargs=("some_args",)
        )
        pool.close()
        pool.join()
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        pool.close()
        pool.join()

    def test_apply(self):
        pool = CtxThreadPool(processes=2)
        assert pool.apply(ctx_func, (1,))[1] == 1
        # for pytest-cov to run on sub processes
        pool.close()
        pool.join()

    def test_apply_async(self):
        pool = CtxThreadPool(processes=2)
        assert pool.apply_async(ctx_func, (1,)).get()[1] == 1
        pool.close()
        pool.join()

    def test_map(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        x = [i for i in range(10)]
        result = pool.map(ctx_func, x)
        assert sorted([r[1] for r in result]) == x
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_map_async(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        x = [i for i in range(10)]
        result = pool.map_async(ctx_func, x).get()
        assert sorted([r[1] for r in result]) == x
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_imap(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        x = [i for i in range(10)]
        result = pool.imap(ctx_func, x)
        assert sorted([r[1] for r in result]) == x
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_imap_unordered(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        x = [i for i in range(10)]
        result = pool.imap_unordered(ctx_func, x)
        assert sorted([r[1] for r in result]) == x
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_starmap(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        xy = [(i, i) for i in range(10)]
        result = pool.starmap(ctx_func2, xy)
        assert sorted([r[1] for r in result]) == [i * 2 for i in range(10)]
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_starmap_async(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        xy = [(i, i) for i in range(10)]
        result = pool.starmap_async(ctx_func2, xy).get()
        assert sorted([r[1] for r in result]) == [i * 2 for i in range(10)]
        assert {r[0] for r in result}.issubset({0, 1})
        pool.close()
        pool.join()

    def test_multiple_ctx_map(self):
        # test whether two context thread pools will interfere with each other
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        pool2 = CtxThreadPool(processes=2, worker_contexts=[2, 3])
        # create enough work items so that the execution period of two pools
        # will overlap
        x = [i for i in range(50000)]
        result = pool.map(ctx_func, x)
        result2 = pool2.map(ctx_func, x)

        assert sorted([r[1] for r in result]) == x
        assert {r[0] for r in result}.issubset({0, 1})
        assert sorted([r[1] for r in result2]) == x
        assert {r[0] for r in result2}.issubset({2, 3})
        pool.close()
        pool.join()
        pool2.close()
        pool2.join()
