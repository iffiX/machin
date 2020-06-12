import os
import time
import torch as t
from datetime import timedelta
from torch.multiprocessing import Process
import torch.distributed as dist

rank = None
messages = 1000
world_size = 4
storage = t.zeros([1000])
storage1 = t.zeros([1000])

def run(i):
    global rank
    rank = i
    dist.init_process_group(backend="gloo",
                            init_method="env://",
                            timeout=timedelta(seconds=2.5),
                            world_size=world_size,
                            rank=rank)

    if i == 0:
        time.sleep(2000)
        print("Process 0 exit.")
        exit(-1)
    t = time.time()
    for j in range(messages):
        for r in range(world_size):
            print("round {}: {}, {} begin".format(j, rank, r))
            try:
                dist.isend(storage, r)
                #dist.recv(storage1)
            except RuntimeError:
                print("round {}: {}, {} error".format(j, rank, r))
            else:
                print("round {}: {}, {}, {:.5f}".format(j, rank, r, time.time() - t))
    print(time.time() - t)
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29530"
    ps = []
    for i in range(world_size):
        p = Process(target=run, args=(i,))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
