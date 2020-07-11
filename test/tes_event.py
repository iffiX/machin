from machin.parallel.event import *
from threading import Thread
import time

event1 = Event()
event2 = Event()
event3 = Event()


def test1():
    global event1, event2, event3
    event = OrEvent(event1, event2, event3)
    event.wait()
    print("hello1")


def test2():
    global event1, event2, event3
    event = AndEvent(AndEvent(event1, event3), event2)
    event.wait()
    print("hello2")


if __name__ == "__main__":
    t1 = Thread(target=test1)
    #t2 = Thread(target=test2)
    t1.start()
    #t2.start()
    time.sleep(0.5)
    print("set event1")
    event1.set()
    print("set event2")
    event2.set()
    print("set event3")
    event3.set()
