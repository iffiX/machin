from machin.parallel.distributed import World
from machin.parallel.server import OrderedServerSimpleImpl
from torch.multiprocessing import spawn
from time import sleep


def main(rank):
    world = World(world_size=3, rank=rank,
                  name=str(rank), rpc_timeout=20)
    # Usually, distributed services in Machin are seperated
    # into:
    # An accessor: OrderedServerSimple
    # An implementation: OrderedServerSimpleImpl
    #
    # Except DistributedBuffer and DistributedPrioritizedBuffer
    #
    # Accessor is a handle, which records the name of internal
    # service handles. Usually paired as an accessible resource
    # to the group, so any group members can get this accessor
    # and use the internal resources & services.
    #
    # Implementation is the thing that actually starts on the
    # provider process, some implementations may contain backend
    # threads, etc.

    group = world.create_rpc_group("group", ["0", "1", "2"])
    if rank == 0:
        _server = OrderedServerSimpleImpl("server", group)
        sleep(5)
    else:
        sleep(2)
        server = group.get_paired("server").to_here()

        # change key "a", set new value to "value"
        # change version from `None` to `1`
        if server.push("a", "value", 1, None):
            print(rank, "push 1 success")
        else:
            print(rank, "push 1 failed")

        # change key "a", set new value to "value2"
        # change version from `1` to `2`
        if server.push("a", "value2", 2, 1):
            print(rank, "push 2 success")
        else:
            print(rank, "push 2 failed")

        # change key "a", set new value to "value3"
        # change version from `2` to `3`
        if server.push("a", "value3", 3, 2):
            print(rank, "push 3 success")
        else:
            print(rank, "push 3 failed")

        assert server.pull("a", None) == ("value3", 3)
        assert server.pull("a", 2) == ("value2", 2)
        assert server.pull("a", 1) is None
        assert server.pull("b", None) is None
        print("Ordered server check passed")
    group.destroy()


if __name__ == "__main__":
    # spawn 3 sub processes
    spawn(main, nprocs=3)
