from machin.parallel.process import Process, ProcessException
import time


def test1():
    time.sleep(1)
    print("Exception occurred at {}".format(time.time()))
    raise RuntimeError("Error")


if __name__ == "__main__":
    t1 = Process(target=test1)
    t1.start()
    while True:
        try:
            t1.watch()
        except ProcessException as e:
            print("Exception caught at {}".format(time.time()))
            print("Exception is: {}".format(e))
            break
    t1.join()
