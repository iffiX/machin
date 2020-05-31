import os
import time
from torch.multiprocessing import Process
import torch.distributed.rpc as rpc

rank = None
messages = 1000
world_size = 4


def test():
    print("Process {} received request.".format(rank))
    time.sleep(1)
    return "SPAM"


def run(i):
    global rank
    rank = i
    if i == world_size - 1:
        time.sleep(4)
        print("Process {} delayed start.".format(i))
    rpc.init_rpc("Rank" + str(i), rank=i, world_size=world_size,
                 rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                     init_method="env://",
                     rpc_timeout=rpc.timedelta(seconds=2),
                     num_send_recv_threads=4))
    if i == 0:
        time.sleep(2)
        print("Process 0 exit.")
        exit(-1)
    t = time.time()
    reqs = []
    for j in range(messages):
        for r in range(world_size):
            reqs.append(rpc.rpc_async("Rank{}".format(r), test, args=()))
        for req, idx in zip(reqs, range(world_size)):
            try:
                print("{} Received from {} : {}".format(rank, idx, req.wait()))
            except RuntimeError:
                print("An error ocurred while {} receiving results from {}".format(rank, idx))
        reqs.clear()
    print(time.time() - t)
    rpc.shutdown(graceful=False)


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
