import os
import time
import torch
import torch.nn as nn
import numpy as np
from threading import Thread
from torch.multiprocessing import Process
import torch.distributed.rpc as rpc

rank = None
messages = 1
world_size = 80


def test(_):
    #return "pong, {}, {}, {}, {}, {}"
    #return "pong, {}, {}, {}, {}, {}".format(rank, rank, rank, rank, rank)
    return None

@torch.jit.script
def test1(_):
    return "pong, {}, {}, {}, {}, {}"


def run(i):
    rpc.init_rpc("Rank"+str(i), rank=i, world_size=world_size,
                 rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                     init_method="env://",
                     rpc_timeout=rpc.timedelta(seconds=60),
                     num_send_recv_threads=4))
    t = time.time()
    reqs = []
    for j in range(messages):
        for r in range(world_size):
            result = rpc.rpc_async("Rank{}".format(r), test,
                                      args=("ping, {}, {}, {}, {}, {}",))
            def wait():
                reqs.append(result.wait())
            Thread(target=wait).start()
    print(time.time() - t)
    rpc.shutdown()


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"
    ps = []
    for i in range(world_size):
        p = Process(target=run, args=(i,))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
