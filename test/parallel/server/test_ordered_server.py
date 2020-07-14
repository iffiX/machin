from machin.parallel.distributed import RpcGroup
from machin.parallel.server import OrderedServerSimple
from test.util_run_multi import *


def _log(rank, msg):
    default_logger.info("Client {}: {}".format(rank, msg))


class TestOrderedServerSimple(WorldTestBase):
    def test__push_pull_service(self):
        fake_group = RpcGroup("fake_group", ("proc1", "proc2"), False)
        fake_group.destroy = lambda: None
        server = OrderedServerSimple("fake_server", "proc1", fake_group)
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
            group = world.create_rpc_group("group", ["0"])
            _server = OrderedServerSimple("o_server", "0", group)
            sleep(5)
        else:
            group = world.get_rpc_group("group", "0")
            server = OrderedServerSimple("o_server", "0", group)

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
        return True
