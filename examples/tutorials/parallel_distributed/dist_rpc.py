from machin.parallel.distributed import World
from torch.multiprocessing import spawn
from time import sleep


# an example service class
class WorkerService:
    counter = 0

    def count(self):
        self.counter += 1
        return self.counter

    def get_count(self):
        return self.counter


def main(rank):
    world = World(world_size=3, rank=rank, name=str(rank), rpc_timeout=20)
    service = WorkerService()
    if rank == 0:
        # only group members needs to enter this function
        group = world.create_rpc_group("group", ["0", "1"])

        service.counter = 20

        # register two services and share a value by pairing
        group.pair("value", service.counter)
        group.register("count", service.count)
        group.register("get_count", service.get_count)

        # cannot register an already used key
        # KeyError will be raised
        # group.register("count", service.count)

        # wait for process 1 to finish
        sleep(4)
        assert service.get_count() == 23

        # deregister service and unpair value
        group.unpair("value")
        group.deregister("count")
        group.deregister("get_count")
        sleep(4)
        group.destroy()
    elif rank == 1:
        group = world.create_rpc_group("group", ["0", "1"])
        sleep(0.5)
        assert group.is_registered("count") and group.is_registered("get_count")
        print("Process 1: service 'count' and 'get_count' correctly " "registered.")

        assert group.registered_sync("count") == 21
        assert group.registered_async("count").wait() == 22
        assert group.registered_remote("count").to_here() == 23
        print("Process 1: service 'count' and 'get_count' correctly " "called")
        sleep(4)
        assert not group.is_registered("count") and not group.is_registered("get_count")
        print("Process 1: service 'count' and 'get_count' correctly " "unregistered.")
        group.destroy()


if __name__ == "__main__":
    # spawn 3 sub processes
    spawn(main, nprocs=3)
