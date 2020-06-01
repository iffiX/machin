from enum import Enum
from time import sleep
from typing import Any, Dict
from datetime import datetime
from queue import Queue, Empty
from torch.distributed import rpc
from abc import ABC, abstractmethod
from threading import Thread, Condition, Lock
from utils.helper_classes import Timer, Counter


class MType(Enum):
    START = 0
    OK = 1
    ALERT = 2
    PING = 3
    PONG = 4


class ElectionGroupBase(ABC):
    def __init__(self):
        self.leader_change_cond = Condition(Lock())

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def get_leader(self):
        pass

    @abstractmethod
    def is_leader(self):
        pass

    def get_leader_change_cond(self):
        return self.leader_change_cond

    @abstractmethod
    def get_members(self):
        pass


class ElectionGroupSimple(ElectionGroupBase):
    def __init__(self, member_ranks, rank, leader_rank=0, *_, **__):
        self.member_ranks = sorted(member_ranks)
        self.leader_rank = leader_rank
        self.rank = rank
        super(ElectionGroupSimple, self).__init__()

    def start(self):
        pass

    def get_leader(self):
        pass

    def is_leader(self):
        return self.rank == self.leader_rank

    def get_members(self):
        return self.member_ranks


class ElectionGroupStableBase(ElectionGroupBase):
    SLEEP_DIVISION = 1000
    elect_groups = {}

    def __init__(self, name, member_ranks, rank, timeout=1, *_, **__):
        super(ElectionGroupStableBase, self).__init__()
        if name in self.elect_groups:
            raise RuntimeError("Election group already existed!")
        self.member_ranks = sorted(member_ranks)
        # relative rank in list
        self.rank = self.member_ranks.index(rank)
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
        self.tka_thread.start()
        self.tto_thread.start()

    def get_leader(self):
        leader = self.leader
        if leader is None:
            return None
        return self.member_ranks[leader]

    def is_leader(self):
        return self.rank == self.leader

    def get_leader_change_cond(self):
        return self.leader_change_cond

    def get_members(self):
        return self.member_ranks

    def _start_round(self, cur_round):
        self._send_all((MType.ALERT, cur_round))
        if self.rank != cur_round % len(self.member_ranks):
            self._send_all((MType.START, cur_round))
        self.cur_round = cur_round
        self.leader = None
        self.ok_counter.reset()
        self.timer.begin()
        self.leader_change_cond.notify_all()

    def _handle(self, timestamp, src, message):
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

                # jump to the nearest available round
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
        while not self.stop:
            if self.rank == self.cur_round % len(self.member_ranks):
                if self.alive_timer.end() >= self.timeout:
                    self._send_all((MType.OK, self.cur_round))
                    self.alive_timer.begin()
            sleep(self.timeout / self.SLEEP_DIVISION)

    @abstractmethod
    def _register_handle(self, handler):
        pass

    @abstractmethod
    def _send(self, to, message):
        pass

    @abstractmethod
    def _send_all(self, message):
        pass

    @staticmethod
    def _timestamp():
        return datetime.utcnow().timestamp()

    def __del__(self):
        self.stop = True
        self.tka_thread.join()
        self.tto_thread.join()


class ElectionGroupStableRpc(ElectionGroupStableBase):
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
        elect_groups = \
            ElectionGroupStableRpc\
            .elect_groups  # type: Dict[str, ElectionGroupStableRpc]
        timeout = elect_groups[group].timeout
        if datetime.utcnow().timestamp() - timestamp <= timeout:
            elect_groups[group]._recv(timestamp, src, message)
