from machin.parallel.distributed import RpcGroup, RoleBase, get_world
from machin.parallel.server import OrderedServerSimple
from ..util_run_multi import *
import pytest


@pytest.fixture(scope="function")
def fake_group():
    group = RpcGroup(("fake_role", 0), "fake_group", [("fake_role", 0)])
    group.destroy = lambda: None
    return group


class OrderedServerRole(RoleBase):
    def __init__(self, index):
        world = get_world()
        super(OrderedServerRole, self).__init__(index)
        self.group = world.create_rpc_group("OServer", roles=[
            ("OServer", 0)
        ])
        self.server = OrderedServerSimple("o_server", "OServer:0", self.group)

    def main(self):
        default_logger.info("Server started")
        sleep(5)
        self.group.destroy()


class ClientRole(RoleBase):
    def __init__(self, index):
        world = get_world()
        super(ClientRole, self).__init__(index)
        self.group = world.get_rpc_group("OServer", "OServer:0")
        self.server = OrderedServerSimple("o_server", "OServer:0", self.group)

    def main(self):
        success = get_world().success
        self._log("started")
        if self.server.push("a", "value", 1, None):
            self._log("push 1 success")
        else:
            self._log("push 1 failed")
        if self.server.push("a", "value2", 2, 1):
            self._log("push 2 success")
        else:
            self._log("push 2 failed")
        if self.server.push("a", "value3", 3, 2):
            self._log("push 3 success")
        else:
            self._log("push 3 failed")

        assert self.server.pull("a", None) == ("value3", 3)
        assert self.server.pull("a", 2) == ("value2", 2)
        assert self.server.pull("a", 1) is None
        assert self.server.pull("b", None) is None
        success[self.role_index] = True

    def _log(self, msg):
        default_logger.info("Client {}: {}".format(self.role_index, msg))


class TestOrderedServerSimple(WorldTestBase):
    @pytest.mark.parametrize("s_role", ["fake_role:0", ("fake_role", 0)])
    def test__push_pull_service(self, s_role, fake_group):
        server = OrderedServerSimple("fake_server", s_role, fake_group)
        assert server._push_service("a", "value", 1, None)
        assert not server._push_service("a", "value1", 2, 0)
        assert server._push_service("a", "value2", 2, 1)
        assert server._push_service("a", "value3", -1, 2)

        assert server._pull_service("a", None) == ("value3", -1)
        assert server._pull_service("a", 2) == ("value2", 2)
        assert server._pull_service("a", 1) is None
        assert server._pull_service("b", None) is None

    @pytest.mark.parametrize("s_role", ["fake_role:0", ("fake_role", 0)])
    def test_push_pull(self, s_role, processes):
        result, watcher = run_multi(processes,
                                    self.subproc_start_world,
                                    args_list=[(
                                        {"OServer": (OrderedServerRole, 1),
                                         "Client": (ClientRole, 3)},
                                    )] * 3)
        watch(processes, watcher)
        default_logger.info("All world inited")
        result, watcher = run_multi(processes,
                                    self.subproc_run_world,
                                    args_list=[(5,)] * 3)
        watch(processes, watcher)
        success = {}
        for r in result:
            success.update(r)
        assert success == {0: True, 1: True, 2: True}
