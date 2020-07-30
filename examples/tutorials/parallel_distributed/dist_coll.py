from machin.parallel.distributed import World
from torch.multiprocessing import spawn
import torch as t


def main(rank):
    world = World(world_size=3, rank=rank,
                  name=str(rank), rpc_timeout=20)
    # all sub processes must enter this function, including non-group members
    group = world.create_collective_group(ranks=[0, 1, 2])
    # test broadcast
    # process 0 will broad cast a tensor filled with 1
    # to process 1 and 2
    if rank == 0:
        a = t.ones([5])
    else:
        a = t.zeros([5])
    group.broadcast(a, 0)
    print(a)
    group.destroy()
    return True


if __name__ == "__main__":
    # spawn 3 sub processes
    spawn(main, nprocs=3)
