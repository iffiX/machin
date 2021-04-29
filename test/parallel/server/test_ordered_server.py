from machin.parallel.server import OrderedServerSimpleImpl
from test.util_run_multi import *
from test.util_platforms import linux_only


def _log(rank, msg):
    default_logger.info(f"Client {rank}: {msg}")


class Object:
    pass


@linux_only
class TestOrderedServerSimple(WorldTestBase):
    def test__push_pull_service(self):
        fake_group = Object()
        fake_group.pair = lambda *_: None
        fake_group.register = lambda *_: None
        fake_group.destroy = lambda: None
        fake_group.is_member = lambda *_: True
        server = OrderedServerSimpleImpl("fake_server", fake_group)
        assert server._push_service("a", "value", 1, None)
        assert not server._push_service("a", "value1", 2, 0)
        assert server._push_service("a", "value2", 2, 1)
        assert server._push_service("a", "value3", -1, 2)

        assert server._pull_service("a", None) == ("value3", -1)
        assert server._pull_service("a", 2) == ("value2", 2)
        assert server._pull_service("a", 1) is None
        assert server._pull_service("b", None) is None

    @staticmethod
    @run_multi(expected_results=[True, True, True])
    @WorldTestBase.setup_world
    def test_push_pull(rank):
        world = get_world()
        if rank == 0:
            group = world.create_rpc_group("group", ["0", "1"])
            _server = OrderedServerSimpleImpl("server", group)
            group.barrier()
            group.barrier()
        elif rank == 1:
            group = world.create_rpc_group("group", ["0", "1"])
            group.barrier()
            server = group.get_paired("server").to_here()

            if server.push("a", "value", 1, None):
                _log(rank, "push 1 success")
            else:
                _log(rank, "push 1 failed")
            if server.push("a", "value2", 2, 1):
                _log(rank, "push 2 success")
            else:
                _log(rank, "push 2 failed")
            if server.push("a", "value3", 3, 2):
                _log(rank, "push 3 success")
            else:
                _log(rank, "push 3 failed")

            assert server.pull("a", None) == ("value3", 3)
            assert server.pull("a", 2) == ("value2", 2)
            assert server.pull("a", 1) is None
            assert server.pull("b", None) is None
            group.barrier()
        return True
