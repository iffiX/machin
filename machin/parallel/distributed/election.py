from enum import Enum
from datetime import datetime
from queue import Queue, Empty
from abc import ABC, abstractmethod
from threading import Event, Lock
from typing import Any, Dict, Union, List

from torch.distributed import rpc
from machin.parallel.thread import Thread
from machin.utils.helper_classes import Timer, Counter
from machin.utils.logging import colorlog, fake_logger


class ElectionGroupBase(ABC):  # pragma: no cover
    """
    Base class for all rpc group election algorithms and their implementations.

    Attributes:
        leader_change_event: An election group should call ``set()`` and then
            ``clear()`` on this event to notify all waiting threads if a leader
            change event has happened. Services relying on the leader may
            create a task thread waiting on this event.
    """

    def __init__(self):
        self.leader_change_event = Event()

    @abstractmethod
    def start(self):
        """
        Start the election algorithm.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the election algorithm.
        """
        pass

    @abstractmethod
    def watch(self):
        """
        Watch for any critical exceptions happening in sub-processes
        and sub-threads started by the election algorithm.
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

    def get_leader_change_event(self):
        """
        Warning:
            This event is allowed to notify the same leader
            change event for indefinite times.

        Returns:
            The leader change event.
        """
        return self.leader_change_event

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

    def stop(self):
        # DOC INHERITED
        pass

    def watch(self):
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
    # Stores references to all stable election groups, rpc calls
    # relies on this dictionary to find the target election group

    # This is better than using a global variable, since its name scope
    # is limited within this class.
    elect_groups = {}

    def __init__(self,
                 name: str,
                 member_ranks: List[int],
                 rank: int,
                 timeout: float = 1,
                 logging: bool = False,
                 **__):
        """
        Args:
            name: Name of this election group.
            member_ranks: Absolute member ranks.
            rank: Rank of the current process.
            timeout: Timeout :math:`\\delta` used in the essay.
        """
        super(ElectionGroupStableBase, self).__init__()
        if name in self.elect_groups:  # pragma: no cover
            raise RuntimeError("Election group with name {} already existed!"
                               .format(name))
        self.member_ranks = sorted(member_ranks)

        # relative rank in list
        self.rank = self.member_ranks.index(rank)

        # Name of group, used to lookup
        self.name = name
        self.elect_groups[name] = self

        self.timer = Timer()
        self.alive_timer = Timer()
        self.timeout_recv_timer = Timer()
        self.timeout_recv_ok_or_start = Event()
        self.timeout_recv_pong = []
        self.timeout_round = -1
        self.timeout = timeout

        self.cur_round = 0
        self.leader = None

        self.lock = Lock()
        self.tka_thread = Thread(target=self._task_keep_alive)
        self.tka_thread.daemon = True
        self.tto_thread = Thread(target=self._task_timeout)
        self.tto_thread.daemon = True
        self.ok_counter = Counter()
        self.last_alert = 0  # from time.time()
        self.last_alert_round = -1

        if logging:
            self.logger = colorlog.getLogger(__name__)
        else:
            self.logger = fake_logger
        self.started = False
        self.run = False

    def start(self):
        # DOC INHERITED
        if not self.started:
            self.run = True
            self.started = True
            self._impl_start()
            # all setup, start round 0
            # sequence is important!
            self._start_round(0)
            self.tka_thread.start()
            self.tto_thread.start()

    def stop(self):
        # DOC INHERITED
        if not self.started:  # pragma: no cover
            raise RuntimeError("Election group not started.")
        else:
            self.run = False
            self.tka_thread.join()
            self.tto_thread.join()
            self._impl_stop()

    def watch(self):
        # DOC INHERITED
        self.tka_thread.watch()
        self.tto_thread.watch()
        self._impl_watch()

    def get_leader(self):
        # DOC INHERITED

        # In order to deal with leader ship changes during
        # member rank lookup, we should fetch the leader first

        # If leadership has changed after fetch, we may use
        # get_leader_change_event to detect the change.
        leader = self.leader
        if leader is not None:
            return self.member_ranks[leader]
        # if there is no leader, return None

    def is_leader(self):
        # DOC INHERITED
        return self.rank == self.leader

    def get_members(self):
        # DOC INHERITED
        return self.member_ranks

    def _impl_start(self):  # pragma: no cover
        # Start the actual implementation class.
        pass

    def _impl_watch(self):  # pragma: no cover
        # Watch for threads/processes started by the
        # actual implementation class.
        pass

    def _impl_stop(self):  # pragma: no cover
        # Stop the actual implementation class.
        pass

    def _start_round(self, cur_round):
        # The start round function defined in essay.
        self._log("start round {}".format(cur_round))

        with self.lock:
            if self.cur_round > cur_round:  # pragma: no cover
                self._log("start round {} failed, higher round {} has started"
                          .format(cur_round, self.cur_round))
                return
            self.timer.begin()
            self.cur_round = cur_round
            self.leader = None
            self.ok_counter.reset()

        # make sure alive messages are sent to all peers immediately
        self.alive_timer._last = 0

        self._send_all((MType.ALERT, cur_round))
        if self.rank != cur_round % len(self.member_ranks):
            self._send_all((MType.START, cur_round))

        self.leader_change_event.set()
        self.leader_change_event.clear()

    def _handle(self, _timestamp, src, message):
        # The handle task defined in essay.
        src_rank = self.member_ranks[src]
        cur_round = self.cur_round
        if message == (MType.OK, cur_round):
            with self.lock:
                if self.cur_round != cur_round:
                    return
                self.ok_counter.count()
                self.timer.begin()
                if self.leader is None and self.ok_counter >= 2:
                    if (self.last_alert_round > cur_round and
                            self._timestamp() - self.last_alert <=
                            6 * self.timeout):
                        return
                    self.leader = cur_round % len(self.member_ranks)
                    self._log("select leader {}".format(self.leader))
                    self.leader_change_event.set()
                    self.leader_change_event.clear()

        elif (message[0] in (MType.OK, MType.START) and
              message[1] > cur_round):
            self._log("move to higher round {}".format(message[1]))
            self._start_round(message[1])
            if message[1] > self.timeout_round:
                self.timeout_recv_ok_or_start.set()

        elif (message[0] in (MType.OK, MType.START) and
              message[1] < cur_round):
            self._log("update peer Process[{}] with lower round {}"
                      .format(src_rank, message[1]))
            self._send(src, (MType.START, cur_round))

        elif (message[0] == MType.ALERT and
              message[1] > cur_round):
            self._log("suspend leader, alerted round {}".format(message[1]))
            with self.lock:
                self.leader = None
                self.last_alert = self._timestamp()
                self.last_alert_round = message[1]
            self.leader_change_event.set()
            self.leader_change_event.clear()

        elif message[0] == MType.PING:
            self._log("respond ping from Process[{}]".format(src_rank))
            self._send(src, (MType.PONG, message[1]))

        elif message == (MType.PONG, self.timeout_round):
            self._log("received pong from Process[{}]".format(src_rank))
            self.timeout_recv_pong.append(src)

    def _task_timeout(self):
        # The Figure 5 optimization in essay.
        while self.run:

            # store current round and leader, so that they will not be changed
            # if a new leader has been elected or a higher round has been
            # notified by other processes during the following process.

            # if the any of the above described event has happened, then the
            # newer round will always >= cur_round
            self.leader_change_event.wait(self.timeout)
            with self.lock:
                cur_round = self.cur_round
                cur_leader_rank = cur_round % len(self.member_ranks)
                cur_passed_time = self.timer.end()

            if cur_passed_time > self.timeout * 2:
                self._log("timeout on leader [{}]".format(cur_leader_rank))
                self._send_all((MType.ALERT, cur_round + 1))

                self.timeout_round = cur_round
                self.timeout_recv_pong.clear()
                self._send_all((MType.PING, cur_round))

                self.timeout_recv_ok_or_start.clear()
                self.timeout_recv_ok_or_start.wait(2 * self.timeout)

                if self.timeout_recv_ok_or_start.is_set():
                    self._log("start higher round / find higher leader")
                    continue

                # Jump to the nearest available round
                # (and the respective leader candidate),
                # skipping all not-responding nodes to save time.
                avail_ranks = sorted(self.timeout_recv_pong.copy())
                self._log("available processes: [{}]".format(avail_ranks))

                for avail_rank in avail_ranks:
                    if avail_rank > cur_leader_rank:
                        new_round = cur_round + (avail_rank - cur_leader_rank)
                        self._log("select process: [{}], start round: {}"
                                  .format(avail_rank, new_round))
                        self._start_round(new_round)
                        break
                else:
                    if avail_ranks:
                        new_round = min(avail_ranks) + \
                                    len(self.member_ranks) - \
                                    cur_leader_rank + cur_round
                        avail_rank = min(avail_ranks)
                        self._log("select process: [{}], start round: {}"
                                  .format(avail_rank, new_round))
                        self._start_round(new_round)
        self._log("timeout task exit normally")

    def _task_keep_alive(self):
        # The keep alive task defined in essay.
        while self.run:
            self.leader_change_event.wait(self.timeout)
            cur_round = self.cur_round
            if self.rank == cur_round % len(self.member_ranks):
                if self.alive_timer.end() >= self.timeout:
                    self._log("notify aliveness")
                    self._send_all((MType.OK, cur_round))
                    self.alive_timer.begin()
        self._log("keep alive task exit normally")

    def _log(self, msg):
        self.logger.info("Stable election group<{}> Process[{}]: "
                         "{}"
                         .format(self.name, self.rank, msg))

    @abstractmethod
    def _send(self, to: int, message: Any):  # pragma: no cover
        # Actual implementations must implement this transmission function.
        # ``to`` is the absolute process rank.
        pass

    @abstractmethod
    def _send_all(self, message: Any):  # pragma: no cover
        # Actual implementations must implement this transmission function.
        pass

    @staticmethod
    def _timestamp():
        return datetime.utcnow().timestamp()


class ElectionGroupStableRpc(ElectionGroupStableBase):
    # Implements the transmission layer using RPC.
    # rpc_sync will throw RuntimeError if target disconnected.
    # rpc_async should will not throw exceptions if target disconnected
    # unless wait() is called, but will execute successfully even
    # without wait().
    def _impl_start(self):
        self.h_thread = Thread(target=self._task_handle)
        self.recv_queue = Queue()
        self.h_thread.start()

    def _impl_stop(self):
        self.h_thread.join()

    def _impl_watch(self):
        self.h_thread.watch()

    def _send(self, to, message):
        ts = self._timestamp()
        to = self.member_ranks[to]
        rpc.rpc_async(str(to), type(self)._reply,
                      args=(ts, self.rank, self.name, message))

    def _send_all(self, message):
        ts = self._timestamp()
        for to in self.member_ranks:
            rpc.rpc_async(str(to), type(self)._reply,
                          args=(ts, self.rank, self.name, message))

    def _task_handle(self):
        while self.run:
            try:
                packet = self.recv_queue.get(block=True, timeout=1e-2)
            except Empty:
                pass
            else:
                self._handle(*packet)
        self._log("handle task exit normally")

    def _recv(self, timestamp, src, message):
        # should only be called by _reply
        self.recv_queue.put_nowait((timestamp, src, message))

    @staticmethod
    def _reply(timestamp: float, src: int, group: str, message: Any):
        # This function is called by rpc api to finish delivering
        # the message.
        elect_groups = \
            ElectionGroupStableRpc \
            .elect_groups  # type: Dict[str, ElectionGroupStableRpc]
        timeout = elect_groups[group].timeout

        # pylint: disable=protected-access
        # Actually it is accessing its own _recv method, not really protected.
        if datetime.utcnow().timestamp() - timestamp <= timeout:
            elect_groups[group]._recv(timestamp, src, message)
