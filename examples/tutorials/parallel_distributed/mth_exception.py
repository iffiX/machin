from machin.parallel.thread import Thread, ThreadException
import time


def test1():
    """
    Test the test1.

    Args:
    """
    time.sleep(1)
    print("Exception occurred at {}".format(time.time()))
    raise RuntimeError("Error")


if __name__ == "__main__":
    t1 = Thread(target=test1)
    t1.start()
    while True:
        try:
            t1.watch()
        except ThreadException as e:
            print("Exception caught at {}".format(time.time()))
            print("Exception is: {}".format(e))
            break
    t1.join()
