from time import sleep
from random import shuffle
from torch.distributed import rpc
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from threading import Event, Lock
from machin.parallel.thread import Thread
from machin.parallel.event import OrEvent
from machin.utils.logging import fake_logger, default_logger


class RoleDispatcherBase(ABC):  # pragma: no cover
    """
    Base role dispatcher.
    """
    @abstractmethod
    def __init__(self,
                 rank: int,
                 world_size: int,
                 roles: List[str],
                 role_counts: List[int],
                 *_, **__):
        """
        Args:
            rank: Absolute current process rank.
            world_size: Size of the whole world.
            roles: A list of role names.
            role_counts: A list of numbers of each role.
        """
        self.has_failure_event = Event()
        self.role_update_event = Event()

    def start(self) -> None:
        """
        Start the role dispatcher.
        """
        pass

    def stop(self) -> None:
        """
        Stop the role dispatcher.
        """
        pass

    def watch(self) -> None:
        """
        Watch for exceptions happening in sub threads/processses
        started by the role dispatcher.
        """
        pass

    @abstractmethod
    def get_roles(self) -> List[Tuple[str, int]]:
        """
        Returns:
            All roles.
        """
        pass

    @abstractmethod
    def get_rank(self, role: Tuple[str, int]) -> Union[int, None]:
        """
        Get the assigned process rank of the specified role.

        Args:
            role: A role tuple like ("some_role", 12), the number indicates
                the role index in its role category.

        Returns:
            Absolute assigned process rank. None if not found.
        """
        pass

    def get_role_update_event(self):
        """
        Returns:
            the role update event.
        """
        return self.role_update_event

    def notify_failure(self, role: Tuple[str, int]) -> None:
        """
        Upper RPC layer will notify role dispatcher the failure of a role,
        dispatcher may reassign this role to other processes immediately,
        or wait for a while, RPC will retry until success if retrying is
        permitted by the user.

        Args:
            role: A role tuple like ("some_role", 12), the number indicates
                the role index in its role category.
        """
        self.has_failure_event.set()
        self.has_failure_event.clear()


class RoleDispatcherSimple(RoleDispatcherBase):
    """
    The simplest static role dispatcher. Requires process number
    be larger or equal to the role number.
    """
    def __init__(self,
                 rank: int,
                 world_size: int,
                 roles: List[str],
                 role_counts: List[int], *_, **__):
        super(RoleDispatcherSimple, self).__init__(
            rank, world_size, roles, role_counts
        )
        # DOC INHERITED
        idx = 0
        role = 0
        for role_count in role_counts:
            idx += role_count
            if idx > rank:
                self.roles = [(roles[role], rank - idx + role_count)]
                break
            role += 1
        else:
            # Extra processes are ignored.
            self.roles = []

        self.all_roles = roles
        self.all_role_counts = role_counts
        # Calculate prefixed role offset so we can lookup
        # the absolute process rank by computing ``role_index + offset``
        self.prefixed_role_count = {}
        count_sum = 0
        for role, role_count in zip(roles, role_counts):
            self.prefixed_role_count[role] = count_sum
            count_sum += role_count

    def start(self):
        self.role_update_event.set()
        self.role_update_event.clear()

    def get_roles(self):
        # DOC INHERITED
        return self.roles

    def get_rank(self, role):
        # DOC INHERITED
        if role[0] in self.all_roles:
            count = self.all_role_counts[self.all_roles.index(role[0])]
            if 0 <= role[1] < count:
                return self.prefixed_role_count[role[0]] + role[1]
        return None


class RoleDispatcherElection(RoleDispatcherBase):
    """
    Role dispatcher based on election group. The election group
    leader wil serve as the dispatcher and allocate roles to processes.
    """
    dispatchers = {}
    logger = fake_logger

    def __init__(self, name, rank,
                 world_size, roles,
                 role_counts,
                 election_group, *_,
                 suspect_threshold=3,
                 reject_threshold=5,
                 dispatch_timeout=1,
                 logging=False, **__):
        super(RoleDispatcherElection, self).__init__(
            rank, world_size, roles, role_counts
        )
        if name in self.dispatchers:  # pragma: no cover
            raise RuntimeError("Dispatcher with name {} already existed!"
                               .format(name))

        self.name = name
        self.rank = rank
        self.dispatchers[name] = self

        self.election_group = election_group

        self.dispatch_thread = Thread(target=self._task_dispatch)

        # This lock is used to control access to the reported failed list
        self.failed_list_lock = Lock()
        self.reported_failed_list = []

        # count for number of failures, when failure reaches a threshold,
        # redistribute the role
        self.suspect_lock = Lock()
        self.suspect = {}
        self.suspect_threshold = suspect_threshold

        # trust table, some processes may fail more oftenly, decrease their
        # trust score by 1/2 for each reported role failure / disconnection
        # when probed
        self.fail_table = {r: 0 for r in range(world_size)}
        self.reject_threshold = reject_threshold

        self.dispatched_roles = []
        self.roles = []
        self.ranks = {}
        self.all_roles = set()
        self.stop_event = Event()
        self.started = False
        # used to deal with the situation where election has completed
        # before starting the dispatcher
        self.first_start = True
        # reserved
        self.dispatch_timeout = dispatch_timeout

        for role, role_count in zip(roles, role_counts):
            for idx in range(role_count):
                self.all_roles.add((role, idx))

        if logging:
            RoleDispatcherElection.logger = default_logger

    def start(self):
        # DOC INHERITED
        if not self.started:
            self.started = True
            self.dispatch_thread.start()
            # in case the election group has not started
            self.election_group.start()

    def stop(self):
        if not self.started:  # pragma: no cover
            raise RuntimeError("Dispatcher not started.")
        else:
            RoleDispatcherElection.dispatchers.pop(self.name)
            self.stop_event.set()
            self.dispatch_thread.join()

    def watch(self):
        self.dispatch_thread.watch()
        self.election_group.watch()

    def get_roles(self):
        # DOC INHERITED
        return self.roles

    def get_rank(self, role):
        # DOC INHERITED
        if role in self.ranks:
            return self.ranks[role]
        else:
            return None

    def notify_failure(self, role: Tuple[str, int]) -> None:
        # DOC INHERITED
        with self.suspect_lock:
            if role in self.suspect:
                self.suspect[role] += 1
            else:
                self.suspect[role] = 1
        self.has_failure_event.set()
        self.has_failure_event.clear()

    def _task_dispatch(self):
        # Wait for leader to change or a failure has happened
        event = OrEvent(self.election_group.get_leader_change_event(),
                        self.has_failure_event,
                        self.stop_event)
        while True:
            if not self.first_start:
                self._log("wait")
                event.wait()
            self._log("wakeup")
            self.first_start = False
            if self.stop_event.is_set():
                break

            failed_roles = []
            self._log("suspect dict: {}".format(self.suspect))
            with self.suspect_lock:
                for k, v in self.suspect.items():
                    if v >= self.suspect_threshold:
                        failed_roles.append(k)
                for r in failed_roles:
                    self.suspect.pop(r)

            with self.failed_list_lock:
                failed_roles += self.reported_failed_list
                self.reported_failed_list.clear()

            if not self.election_group.is_leader() and failed_roles:
                try:
                    rpc.rpc_sync(str(self.election_group.get_leader()),
                                 RoleDispatcherElection.
                                 _rpc_report_failed_roles,
                                 args=(self.name, failed_roles))
                    self._log("reported failures: {}".format(failed_roles))
                except RuntimeError:
                    continue

            while self.election_group.is_leader():  # pragma: no cover
                if self._assign_roles(failed_roles):
                    if self._update_ranks():
                        break
                if self.stop_event.is_set():
                    return
                sleep(0.1)

    def _assign_roles(self, failed_roles):
        future = []
        assigned_map = {}
        assigned_roles = set()
        available_members = []
        unassigned_members = []
        self._log("begin dispatch roles")

        for fr in failed_roles:
            executor = self.get_rank(fr)
            if executor is not None:
                self.fail_table[executor] += 2

        # Get assigned roles from all processes, also test their aliveness.
        for member in self.election_group.get_members():
            # TODO: add timeout
            future.append((
                member,
                rpc.rpc_async(str(member),
                              RoleDispatcherElection._rpc_respond_roles,
                              args=(self.name,))
            ))
        for member, fut in future:
            try:
                member_roles = fut.wait()
                assigned_roles.update(member_roles)
                assigned_map[member] = member_roles
                if len(member_roles) == 0:
                    unassigned_members.append(member)
                available_members.append(member)
            except RuntimeError:  # pragma: no cover
                # timeout, peer reset (disconnect) etc.
                self.fail_table[member] += 1

        # Assign unassigned roles.
        unassigned_roles = list(self.all_roles -
                                (assigned_roles - set(failed_roles)))

        # sort unassigned_members and available_members according
        # to their score and rank
        unassigned_members = self._sort_members(unassigned_members,
                                                self.fail_table,
                                                self.reject_threshold)
        available_members = self._sort_members(available_members,
                                               self.fail_table,
                                               self.reject_threshold)

        self._log("available members: {}".format(available_members))
        self._log("unassigned members: {}".format(unassigned_members))
        self._log("assigned roles: {}".format(assigned_roles))
        self._log("failed roles: {}".format(failed_roles))
        self._log("fail table: {}".format(self.fail_table))

        assign_map = self._plan_assign_roles(unassigned_roles,
                                             unassigned_members,
                                             available_members)
        self._log("assign map for floating roles: {}".format(assign_map))

        if assign_map is None:  # pragma: no cover
            raise RuntimeError("No available members to assign!")

        future.clear()
        success = []

        # update old assigned map with the new assign map
        # remove revoked roles
        for member, old_roles in assigned_map.items():
            new_roles = list(set(old_roles) - set(failed_roles))
            if member in assign_map:
                new_roles += assign_map[member]
            old_roles.clear()
            old_roles += new_roles
        self._log("new assign map: {}".format(assigned_map))

        # Dispatch unassigned roles with original roles to all processes.
        for member, roles in assigned_map.items():
            # TODO: add timeout
            future.append((
                member,
                rpc.rpc_async(
                    str(member),
                    RoleDispatcherElection._rpc_dispatch_roles,
                    args=(self.name, roles,)
                )
            ))
        for member, fut in future:
            try:
                success.append(fut.wait())
            except RuntimeError:  # pragma: no cover
                # timeout, peer reset (disconnect) etc.
                # Failed to assign, restart assign process.
                success.append(False)
                self.fail_table[member] += 1

        dispatch_result = {rank: res
                           for rank, res in zip(assigned_map.keys(), success)}
        self._log("dispatch success result: {}".format(dispatch_result))
        return all(success)

    def _update_ranks(self):
        self._log("begin update ranks")
        future = []
        # Get assigned roles from all processes, also test their aliveness.
        for member in self.election_group.get_members():
            # TODO: add timeout
            future.append((
                member,
                rpc.rpc_async(str(member),
                              RoleDispatcherElection._rpc_respond_roles,
                              args=(self.name,))
            ))

        for member, fut in future:
            try:
                member_roles = fut.wait()
                for member_role in member_roles:
                    self.ranks[member_role] = member
            except RuntimeError:  # pragma: no cover
                # timeout, peer reset (disconnect) etc.
                self.fail_table[member] += 1

        self._log("new rank table: {}".format(self.ranks))
        if set(self.ranks.keys()) != set(self.all_roles):  # pragma: no cover
            self._log("some roles are not assigned, begin retry")
            return False
        future.clear()
        # Push role-rank route table to all processes.
        for member in self.election_group.get_members():
            # TODO: add timeout
            future.append((
                member,
                rpc.rpc_async(str(member),
                              RoleDispatcherElection._rpc_set_ranks,
                              args=(self.name, self.ranks,))
            ))
        for member, fut in future:
            try:
                if not fut.wait():  # pragma: no cover
                    self._log("Failed to push rank table pushed to {},"
                              "group not exist"
                              .format(member))
            except RuntimeError:  # pragma: no cover
                # timeout, peer reset (disconnect) etc.
                self._log("Failed to push rank table pushed to {}, "
                          "disconnected"
                          .format(member))
                self.fail_table[member] += 1
        self._log("rank table pushed to all alive members successfully!")
        return True

    def _log(self, msg):
        self.logger.info("Election dispatcher<{}> Process[{}]: "
                         "{}"
                         .format(self.name, self.rank, msg))

    @staticmethod
    def _sort_members(members, fail_table, reject_threshold):
        m_list = [(fail_table[m], m) for m in members
                  if fail_table[m] < reject_threshold]
        # less failure first
        m_list = sorted(m_list)
        return [m[1] for m in m_list]

    @staticmethod
    def _plan_assign_roles(roles, unassigned_members, available_members):
        # Decide how to assign unassigned roles.
        shuffle(available_members)
        result = {member: [] for member in available_members}

        # assign roles to free members first
        idx = 0
        if not available_members:  # pragma: no cover
            return None
        for role in roles:
            if len(unassigned_members) > 0:
                result[unassigned_members[0]].append(role)
                unassigned_members.pop(0)
            else:
                # Assign remaining roles in a round robin fashion.
                result[available_members[idx]].append(role)
                idx = (idx + 1) % len(available_members)
        return result

    @classmethod
    def _rpc_respond_roles(cls, name):
        if name not in cls.dispatchers:  # pragma: no cover
            return []
        return cls.dispatchers[name].roles

    @classmethod
    def _rpc_dispatch_roles(cls, name, roles):
        if name not in cls.dispatchers:  # pragma: no cover
            return False
        dispatcher = cls.dispatchers[name]
        dispatcher.roles = roles
        return True

    @classmethod
    def _rpc_set_ranks(cls, name, new_ranks):
        if name not in cls.dispatchers:  # pragma: no cover
            return False
        dispatcher = cls.dispatchers[name]
        dispatcher.ranks = new_ranks
        dispatcher.role_update_event.set()
        dispatcher.role_update_event.clear()
        return True

    @classmethod
    def _rpc_report_failed_roles(cls, name, failed_roles):
        if name not in cls.dispatchers:  # pragma: no cover
            return False
        dispatcher = cls.dispatchers[name]
        with dispatcher.failed_list_lock:
            dispatcher.reported_failed_list = failed_roles
        dispatcher.has_failure_event.set()
        dispatcher.has_failure_event.clear()
        return True
