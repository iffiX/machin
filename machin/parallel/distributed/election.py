from enum import Enum
from time import sleep
from datetime import datetime
from queue import Queue, Empty
from abc import ABC, abstractmethod
from threading import Thread, Condition, Lock
from typing import Any, Dict, Union, List, Callable

from torch.distributed import rpc
from machin.utils.helper_classes import Timer, Counter


class ElectionGroupBase(ABC):
    """
    Base class for all rpc group election algorithms and their implementations.

    Attributes:
        leader_change_cond: An election group should call ``notify_all()`` on
            this conditional if a leader change event has happened. Services
            relying on the leader may create a task thread waiting on this
            conditional.
    """
    def __init__(self):
        self.leader_change_cond = Condition(Lock())

    @abstractmethod
    def start(self):
        """
        Start the election algorithm.
        """
        pass

    @abstractmethod
    def get_leader(self) -> Union[int, None]:
        """
        Returns:
            Absolute global process rank of the current leader.

            If no leader has been elected, return ``None``.
        """
        pass

    @abstractmethod
    def is_leader(self) -> bool:
        """
        Returns:
            Whether current process is the leader.
        """
        pass

    def get_leader_change_cond(self):
        """
        Warning:
            This conditional is allowed to notify the same leader
            change event for indefinite times.

        Returns:
            The leader change conditional.
        """
        return self.leader_change_cond

    @abstractmethod
    def get_members(self) -> List[int]:
        """
        Returns:
            A list of the absolute global rank of all current members.
        """
        pass


class ElectionGroupSimple(ElectionGroupBase):
    """
    The simplest static election group, leader is specified on group creation.
    And will remain the same in the whole life time of the group. Requires no
    communication and adds zero overhead.
    """
    def __init__(self,
                 member_ranks: List[int],
                 rank: int,
                 leader_rank: int = 0,
                 **__):
        """
        Args:
            member_ranks: Absolute member process ranks.
            rank: Rank of the current process.
            leader_rank: Absolute leader process rank
        """
        self.member_ranks = sorted(member_ranks)
        self.leader_rank = leader_rank
        self.rank = rank
        super(ElectionGroupSimple, self).__init__()

    def start(self):
        # DOC INHERITED
        pass

    def get_leader(self):
        # DOC INHERITED
        return self.leader_rank

    def is_leader(self):
        # DOC INHERITED
        return self.rank == self.leader_rank

    def get_members(self):
        # DOC INHERITED
        return self.member_ranks


class MType(Enum):
    """
    Message types used in
    :class:`~machin.parallel.distributed.election.ElectionGroupStableBase`
    """
    START = 0
    OK = 1
    ALERT = 2
    PING = 3
    PONG = 4


class ElectionGroupStableBase(ElectionGroupBase):
    """
    This class implements the stable election algorithm described in:
    `<<Stable leader election>> <http://citeseerx.ist.psu.edu/viewdoc/\
download?doi=10.1.1.89.4817&rep=rep1&type=pdf>`_

    The essay stated that the algorithm can establish leadership within
    :math:`9\\delta` time. The algorithm requires :math:`O(n^2)` messages
    to elect and :math:`O(n)` messages to maintain. But the leadership
    is stable, so no scenarios like leadership switching back and forth
    between two nodes will ocurr, this is favorable.

    Code in this class implements the algorithm described in Figure 4 with
    optimizations in Figure 5 of the essay.
    """
    # The ratio of timeout value to sleep time between probes.
    SLEEP_DIVISION = 1000

    # Stores references to all stable election groups, rpc calls
    # relies on this dictionary to find the target election group/

    # This is better than using a global variable, since its name scope
    # is limited within this class.
    elect_groups = {}

    def __init__(self,
                 name: str,
                 member_ranks: List[int],
                 rank: int,
                 timeout: float = 1,
                 **__):
        """
        Args:
            name: Name of this election group.
            member_ranks: Absolute member ranks.
            rank: Rank of the current process.
            timeout: Timeout :math:`\\delta` used in the essay.
        """
        super(ElectionGroupStableBase, self).__init__()
        if name in self.elect_groups:
            raise RuntimeError("Election group already existed!")
        self.member_ranks = sorted(member_ranks)

        # relative rank in list
        self.rank = self.member_ranks.index(rank)
        # Name of group, used to lookup
        self.name = name

        self.timer = Timer()
        self.alive_timer = Timer()
        self.timeout_recv_timer = Timer()
        self.timeout_recv_ok_or_start = False
        self.timeout_recv_pong = []
        self.timeout = timeout

        self.cur_round = 0
        self.leader = None

        self.tka_thread = Thread(target=self._task_keep_alive)
        self.tto_thread = Thread(target=self._task_timeout)
        self.leader_change_cond = Condition(Lock())
        self.ok_counter = Counter()
        self.last_alert = None
        self.recv_alerts = []
        self.elect_groups[name] = self

        self.stop = False
        self.tka_thread.start()
        self.tto_thread.start()
        self._start_round(0)
        self._register_handle(self._handle)

    def start(self):
        # DOC INHERITED
        self.tka_thread.start()
        self.tto_thread.start()

    def get_leader(self):
        # DOC INHERITED

        # In order to deal with leader ship changes during
        # member rank lookup, we should fetch the leader first

        # If leadership has changed after fetch, we may use
        # leader_change_cond to detect the change.
        leader = self.leader
        if leader is None:
            return None
        return self.member_ranks[leader]

    def is_leader(self):
        # DOC INHERITED
        return self.rank == self.leader

    def get_members(self):
        # DOC INHERITED
        return self.member_ranks

    def _start_round(self, cur_round):
        # The start round function defined in essay.
        self._send_all((MType.ALERT, cur_round))
        if self.rank != cur_round % len(self.member_ranks):
            self._send_all((MType.START, cur_round))
        self.cur_round = cur_round
        self.leader = None
        self.ok_counter.reset()
        self.timer.begin()
        self.leader_change_cond.notify_all()

    def _handle(self, timestamp, src, message):
        # The handle task defined in essay.
        if message == (MType.OK, self.cur_round):
            self.ok_counter.count()
            if (self.leader is None and
                    self.ok_counter >= 2 and
                    (self.last_alert is None or
                     self._timestamp() - timestamp > 6 * self.timeout)):
                self.leader = self.cur_round % len(self.member_ranks)
                self.timer.begin()
                self.leader_change_cond.notify_all()

        elif (message[0] in (MType.OK, MType.START) and
              message[1] > self.cur_round):
            self._start_round(self.cur_round + 1)
            self.timeout_recv_ok_or_start = True

        elif (message[0] in (MType.OK, MType.START) and
              message[1] < self.cur_round):
            self._send(src, (MType.START, self.cur_round))

        elif (message[0] == MType.ALERT and
              message[1] > self.cur_round):
            self.leader = None
            self.last_alert = self._timestamp()
            self.recv_alerts.append((timestamp, message))
            self.leader_change_cond.notify_all()

        elif message[0] == MType.PING:
            self._send(src, (MType.PONG, message[1]))

        elif message == (MType.PONG, self.cur_round):
            self.timeout_recv_pong.append(src)

    def _task_timeout(self):
        # The Figure 5 optimization in essay.
        while not self.stop:
            if self.timer.end() > self.timeout * 2:
                self._send_all((MType.ALERT, self.cur_round + 1))

                self.timeout_recv_pong.clear()
                self._send_all((MType.PING, self.cur_round))

                self.timeout_recv_ok_or_start = False
                self.timeout_recv_timer.begin()

                while (not self.timeout_recv_ok_or_start and
                       self.timeout_recv_timer.end() <= 2 * self.timeout):
                    sleep(self.timeout / self.SLEEP_DIVISION)
                if self.timeout_recv_ok_or_start:
                    continue

                # Jump to the nearest available round
                # (and the respective leader candidate),
                # skipping all not-responding nodes to save time.
                cur_leader_rank = self.cur_round % len(self.member_ranks)
                avail_ranks = self.timeout_recv_pong.copy()
                for avail_rank in avail_ranks:
                    if avail_rank > cur_leader_rank:
                        new_round = self.cur_round + \
                                    (avail_rank - cur_leader_rank)
                        self._start_round(new_round)
                        break
                else:
                    new_round = min(avail_ranks) + len(self.member_ranks) - \
                                cur_leader_rank + self.cur_round
                    self._start_round(new_round)
            else:
                sleep(self.timeout / self.SLEEP_DIVISION)

    def _task_keep_alive(self):
        # The keep alive task defined in essay.
        while not self.stop:
            if self.rank == self.cur_round % len(self.member_ranks):
                if self.alive_timer.end() >= self.timeout:
                    self._send_all((MType.OK, self.cur_round))
                    self.alive_timer.begin()
            sleep(self.timeout / self.SLEEP_DIVISION)

    @abstractmethod
    def _register_handle(self, handler: Callable):
        # Register the _handle() function, since some communication
        # libraries use callback handlers to deal with incoming messages.
        pass

    @abstractmethod
    def _send(self, to: int, message: Any):
        # Actual implementations must implement this transmission function.
        # ``to`` is the absolute process rank.
        pass

    @abstractmethod
    def _send_all(self, message: Any):
        # Actual implementations must implement this transmission function.
        pass

    @staticmethod
    def _timestamp():
        return datetime.utcnow().timestamp()

    def __del__(self):
        # Close this group.
        self.stop = True
        self.tka_thread.join()
        self.tto_thread.join()


class ElectionGroupStableRpc(ElectionGroupStableBase):
    # Implements the transmission layer using RPC.
    def _register_handle(self, handler):
        self.h_thread = Thread(target=self._task_handle)
        self.recv_queue = Queue()

    def _send(self, to, message):
        ts = self._timestamp()
        to = self.member_ranks[to]
        rpc.rpc_async(str(to), self._reply,
                      args=(ts, self.rank, self.name, message))

    def _send_all(self, message):
        ts = self._timestamp()
        for to in self.member_ranks:
            rpc.rpc_async(str(to), self._reply,
                          args=(ts, self.rank, self.name, message))

    def _task_handle(self):
        while not self.stop:
            try:
                packet = self.recv_queue.get(block=True, timeout=1e-3)
            except Empty:
                pass
            else:
                self._handle(*packet)

    def _recv(self, timestamp, src, message):
        # should only be called by _reply
        self.recv_queue.put_nowait((timestamp, src, message))

    @staticmethod
    def _reply(timestamp: float, src: int, group: str, message: Any):
        # This function is called by rpc api to finish delivering
        # the message.
        elect_groups = \
            ElectionGroupStableRpc\
            .elect_groups  # type: Dict[str, ElectionGroupStableRpc]
        timeout = elect_groups[group].timeout

        # pylint: disable=protected-access
        # Actually it is accessing its own _recv method, not really protected.
        if datetime.utcnow().timestamp() - timestamp <= timeout:
            elect_groups[group]._recv(timestamp, src, message)
