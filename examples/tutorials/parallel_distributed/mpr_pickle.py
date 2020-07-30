from machin.parallel.pickle import dumps, loads
from machin.parallel.process import Process
from machin.parallel import get_context
import torch as t


def print_tensor_sub_proc(tens):
    print(loads(tens))


def exec_sub_proc(func):
    loads(func)()


if __name__ == "__main__":
    spawn_ctx = get_context("spawn")
    fork_ctx = get_context("fork")
    # cpu tensor, not in shared memory
    # If you would like to pass this tensor to a sub process
    # set copy_tensor to `True`, otherwise only a pointer to
    # memory will be passed to the subprocess.
    # However, if you do this in the same process, no SEGFAULT
    # will happen, because memory map is the same.
    tensor = t.ones([10])
    p = Process(target=print_tensor_sub_proc,
                args=(dumps(tensor, copy_tensor=True),),
                ctx=fork_ctx)
    p.start()
    p.join()
    # cpu tensor, in shared memory

    # If you would like to pass this tensor to a sub process
    # set copy_tensor to `False` is more efficient, because
    # only a pointer to the shared memory will be passed, and
    # not all data in the tensor.
    tensor.share_memory_()
    p = Process(target=print_tensor_sub_proc,
                args=(dumps(tensor, copy_tensor=False),),
                ctx=fork_ctx)
    p.start()
    p.join()
    print("Dumped length of shm tensor if copy: {}"
          .format(len(dumps(tensor, copy_tensor=True))))
    print("Dumped length of shm tensor if not copy: {}"
          .format(len(dumps(tensor, copy_tensor=False))))

    # gpu tensor
    # If you would like to pass this tensor to a sub process
    # set copy_tensor to `False` is more efficient, because
    # only a pointer to the CUDA memory will be passed, and
    # not all data in the tensor.
    # You should use "spawn" context instead of "fork" as well.
    tensor = tensor.to("cuda:0")
    p = Process(target=print_tensor_sub_proc,
                args=(dumps(tensor, copy_tensor=False),),
                ctx=spawn_ctx)
    p.start()
    p.join()
    print("Dumped length of gpu tensor if copy: {}"
          .format(len(dumps(tensor, copy_tensor=True))))
    print("Dumped length of gpu tensor if not copy: {}"
          .format(len(dumps(tensor, copy_tensor=False))))

    # in order to pass a local function / lambda function
    # to the subprocess, set recursive to `true`
    # then refered nonlocal&global variable will also be serialized.
    def local_func():
        global tensor
        tensor.fill_(3)


    print("Before:{}".format(tensor))
    p = Process(target=exec_sub_proc,
                args=(dumps(local_func, recurse=True, copy_tensor=False),),
                ctx=spawn_ctx)
    p.start()
    p.join()
    print("After:{}".format(tensor))
