from machin.parallel.process import Process, ProcessException
import time


def test1():
    time.sleep(1)
    print(f"Exception occurred at {time.time()}")
    raise RuntimeError("Error")


if __name__ == "__main__":
    t1 = Process(target=test1)
    t1.start()
    while True:
        try:
            t1.watch()
        except ProcessException as e:
            print(f"Exception caught at {time.time()}")
            print(f"Exception is: {e}")
            break
    t1.join()
