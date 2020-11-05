from machin.parallel.event import *
from machin.parallel.thread import Thread
import time

event1 = Event()
event2 = Event()
event3 = Event()

# wait() will block until its value might have changed (due to a sub event)
# wait() returns a bool value


def test1():
    """
    Perform the first event of the same as a function.

    Args:
    """
    global event1, event2, event3
    event = OrEvent(event1, event2, event3)
    while not event.wait():
        continue
    # will print if any one of these events are set
    print("hello1")


def test2():
    """
    Perform a single event.

    Args:
    """
    global event1, event2, event3
    event = AndEvent(AndEvent(event1, event3), event2)
    while not event.wait():
        continue
    # will print if event1, event2 and event3 are all set
    print("hello2")


if __name__ == "__main__":
    t1 = Thread(target=test1)
    t2 = Thread(target=test2)
    t1.start()
    t2.start()
    print("set event1")
    event1.set()

    time.sleep(1)
    print("set event2")
    event2.set()
    print("set event3")
    event3.set()
