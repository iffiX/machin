from machin.parallel.pool import (
    Pool,
    CtxPool,
    ThreadPool,
    CtxThreadPool
)
import dill
import pytest
import torch as t


def init_func(*_):
    print("Hello")


def func(x):
    return t.sum(x * 2)


def func2(x, y):
    return t.sum(x + y)


class TestPool(object):
    def test_apply(self):
        pool = Pool(processes=2)
        x = t.ones([10])
        assert pool.apply(func, (x,)) == 20

        # for pytest-cov to run on sub processes
        pool.close()
        pool.join()

    def test_apply_async(self):
        pool = Pool(processes=2)
        x = t.ones([10])
        assert pool.apply_async(func, (x,)).get() == 20
        pool.close()
        pool.join()

    def test_map(self):
        pool = Pool(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map(func, x),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_map_async(self):
        pool = Pool(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map_async(func, x).get(),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_imap(self):
        pool = Pool(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.imap(func, x),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_imap_unordered(self):
        pool = Pool(processes=2)
        x = [t.ones([10]) * i for i in range(5)]
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       sorted(pool.imap_unordered(func, x)),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_starmap(self):
        pool = Pool(processes=2)
        x = [(t.ones([10]) * i, t.ones([10]) * i)
             for i in range(5)]
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.starmap(func2, x),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_starmap_async(self):
        pool = Pool(processes=2)
        x = [(t.ones([10]) * i, t.ones([10]) * i)
             for i in range(5)]
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.starmap_async(func2, x).get(),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_gpu_tensor(self, pytestconfig):
        x = [t.ones([10], device=pytestconfig.getoption("gpu_device")) * i
             for i in range(5)]

        pool = Pool(processes=2, is_copy_tensor=True)
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map(func, x),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

        pool = Pool(processes=2, is_copy_tensor=False, share_method="cuda")
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map(func, x),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_cpu_shared_tensor(self):
        x = [t.ones([10]) * i for i in range(5)]
        for xx in x:
            xx.share_memory_()
        pool = Pool(processes=2, is_copy_tensor=False, share_method="cpu")
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map(func, x),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_lambda_and_local(self):
        x = [t.ones([10]) * i for i in range(5)]
        y = t.ones([10])

        x2 = [(t.ones([10]) * i, t.ones([10]) * i)
              for i in range(5)]

        def local_func(xx):
            nonlocal y
            return t.sum(xx + y)

        pool = Pool(processes=2, is_recursive=True)
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map(local_func, x),
                       [10, 20, 30, 40, 50]
                   ))
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map(lambda xx: t.sum(xx[0] + xx[1]), x2),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

        pool = Pool(processes=2)
        assert all(out == expect_out for
                   out, expect_out in
                   zip(
                       pool.map(func, x),
                       [0, 20, 40, 60, 80]
                   ))
        pool.close()
        pool.join()

    def test_size(self):
        pool = Pool(processes=2)
        assert pool.size() == 2
        pool.close()
        pool.join()

    def test_reduce(self):
        with pytest.raises(RuntimeError, match="not reducible"):
            dill.dumps(Pool(processes=2))


def ctx_func(ctx, x):
    # pretend to have done something using x wih context ctx
    return ctx, x


def ctx_func2(ctx, x, y):
    # pretend to have done something using x and y wwih context ctx
    return ctx, x + y


class TestCtxPool(object):
    def test_init(self):
        with pytest.raises(ValueError, match="not equal to the number"):
            _ = CtxPool(processes=2, worker_contexts=[0, 1, 2, 3])
        pool = CtxPool(processes=2, initializer=init_func,
                       initargs=("some_args",))
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
        assert set([r[0] for r in result]).issubset({0, 1})
        pool.close()
        pool.join()

    def test_ctx_starmap(self):
        pool = CtxPool(processes=2, worker_contexts=[0, 1])
        # make sure both workers will have items
        xy = [(i, i) for i in range(10)]
        result = pool.starmap(ctx_func2, xy)
        assert sorted([r[1] for r in result]) == [i * 2 for i in range(10)]
        assert set([r[0] for r in result]).issubset({0, 1})
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
        assert set([r[0] for r in result]).issubset({0, 1})
        assert sorted([r[1] for r in result2]) == x
        assert set([r[0] for r in result2]).issubset({2, 3})
        pool.close()
        pool.join()
        pool2.close()
        pool2.join()


class TestThreadPool(object):
    def test_size(self):
        pool = ThreadPool(processes=2)
        assert pool.size() == 2
        pool.close()
        pool.join()

    def test_reduce(self):
        with pytest.raises(RuntimeError, match="not reducible"):
            dill.dumps(ThreadPool(processes=2))


class TestCtxThreadPool(object):
    def test_init(self):
        with pytest.raises(ValueError, match="not equal to the number"):
            _ = CtxThreadPool(processes=2, worker_contexts=[0, 1, 2, 3])
        pool = CtxThreadPool(processes=2, initializer=init_func,
                             initargs=("some_args",))
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
        assert set([r[0] for r in result]).issubset({0, 1})
        pool.close()
        pool.join()

    def test_map_async(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        x = [i for i in range(10)]
        result = pool.map_async(ctx_func, x).get()
        assert sorted([r[1] for r in result]) == x
        assert set([r[0] for r in result]).issubset({0, 1})
        pool.close()
        pool.join()

    def test_imap(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        x = [i for i in range(10)]
        result = pool.imap(ctx_func, x)
        assert sorted([r[1] for r in result]) == x
        assert set([r[0] for r in result]).issubset({0, 1})
        pool.close()
        pool.join()

    def test_imap_unordered(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        x = [i for i in range(10)]
        result = pool.imap_unordered(ctx_func, x)
        assert sorted([r[1] for r in result]) == x
        assert set([r[0] for r in result]).issubset({0, 1})
        pool.close()
        pool.join()

    def test_starmap(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        xy = [(i, i) for i in range(10)]
        result = pool.starmap(ctx_func2, xy)
        assert sorted([r[1] for r in result]) == [i * 2 for i in range(10)]
        assert set([r[0] for r in result]).issubset({0, 1})
        pool.close()
        pool.join()

    def test_starmap_async(self):
        pool = CtxThreadPool(processes=2, worker_contexts=[0, 1])
        xy = [(i, i) for i in range(10)]
        result = pool.starmap_async(ctx_func2, xy).get()
        assert sorted([r[1] for r in result]) == [i * 2 for i in range(10)]
        assert set([r[0] for r in result]).issubset({0, 1})
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
        assert set([r[0] for r in result]).issubset({0, 1})
        assert sorted([r[1] for r in result2]) == x
        assert set([r[0] for r in result2]).issubset({2, 3})
        pool.close()
        pool.join()
        pool2.close()
        pool2.join()
