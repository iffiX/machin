from time import sleep
from random import shuffle
from torch.distributed import rpc
from abc import ABC, abstractmethod
from typing import List, Tuple
from threading import Thread, Condition, Lock


class RoleDispatcherBase(ABC):
    """
    Base role dispatcher.
    """
    @abstractmethod
    def __init__(self,
                 rank: int,
                 world_size: int,
                 roles: List[str],
                 role_counts: List[int]):
        """
        Args:
            rank: Absolute current process rank.
            world_size: Size of the whole world.
            roles: A list of role names.
            role_counts: A list of numbers of each role.
        """
        def f(*_):
            pass
        # Prevent pylint from complaining
        f(rank, world_size, roles, role_counts)
        self.has_failure_cond = Condition(Lock())
        self.role_update_cond = Condition(Lock())

    @abstractmethod
    def start(self) -> None:
        """
        Start the role dispatcher.
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
    def get_rank(self, role: Tuple[str, int]) -> int:
        """
        Get the assigned process rank of the specified role.

        Args:
            role: A role tuple like ("some_role", 12), the number indicates
                the role index in its role category.

        Returns:
            Absolute assigned process rank.
        """
        pass

    def get_role_update_cond(self):
        """
        Returns:
            the role update conditional.
        """
        return self.role_update_cond

    @abstractmethod
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
        pass


class RoleDispatcherSimple(RoleDispatcherBase):
    """
    The simplest static role dispatcher. Requires process number
    be larger or equal to the role number.
    """
    def __init__(self, rank, world_size, roles, role_counts):
        # DOC INHERITED
        idx = 0
        role = 0
        for role_count in role_counts:
            idx += role_count
            if idx > rank:
                self.roles = [roles[role]]
                break
            role += 1
        else:
            # Extra processes are ignored.
            self.role = []

        # Calculate prefixed role offset so we can lookup
        # the absolute process rank by computing ``role_index + offset``
        self.prefixed_role_count = {}
        count_sum = 0
        for role, role_count in zip(roles, role_counts):
            self.prefixed_role_count[role] = count_sum
            count_sum += role_count

        super(RoleDispatcherSimple, self).__init__(
            rank, world_size, roles, role_counts
        )

    def start(self):
        # DOC INHERITED
        pass

    def get_roles(self):
        # DOC INHERITED
        return self.roles

    def get_rank(self, role):
        # DOC INHERITED
        return self.prefixed_role_count[role[0]] + role[1]

    def notify_failure(self, role: Tuple[str, int]) -> None:
        # DOC INHERITED
        pass


class RoleDispatcherElection(RoleDispatcherBase):
    """
    Role dispatcher based on election group. The election group
    leader wil serve as the dispatcher and allocate roles to processes.
    """
    def __init__(self, rank, world_size, role_names, role_nums,
                 election_group):
        self.elect_group = election_group

        self.lead_thread = Thread(target=self._task_lead_group)
        self.deal_fail_thread = Thread(target=self._task_deal_failure)
        # This lock is used to prevent the lead thread and deal_fail
        # thread both execute the reassign task.
        self.assign_lock = Lock()

        self.dispatched_roles = []
        self.roles = []
        self.ranks = {}
        self.all_roles = set()
        for role, role_count in zip(role_names, role_nums):
            for idx in range(role):
                self.all_roles.add((role, idx))

        super(RoleDispatcherElection, self).__init__(
            rank, world_size, role_names, role_nums
        )

    def start(self):
        # DOC INHERITED
        self.lead_thread.start()
        self.deal_fail_thread.start()
        # Elect thread must be started before the elect group to
        # let it wait on the leader change signal.
        self.elect_group.start()

    def get_roles(self):
        # DOC INHERITED
        return self.roles

    def get_rank(self, role):
        # DOC INHERITED
        return self.ranks[role]

    def notify_failure(self, role):
        # DOC INHERITED
        self.has_failure_cond.notify_all()

    def _task_lead_group(self):
        # Wait for leader to change
        self.elect_group.get_leader_change_cond().wait()
        self.assign_lock.acquire()
        while (self.elect_group.is_leader()
               and not self._assign_roles()
               and not self._update_ranks()):
            sleep(0.1)
        self.assign_lock.release()

    def _task_deal_failure(self):
        # Deal with failure
        self.has_failure_cond.wait()
        self.assign_lock.acquire()
        while (self.elect_group.is_leader()
               and not self._assign_roles()
               and not self._update_ranks()):
            sleep(0.1)
        self.assign_lock.release()

    def _assign_roles(self):
        future = []
        assigned_roles = set()
        available_members = []
        unassigned_members = []
        # Get assigned roles from all processes, also test their aliveness.
        for member in self.elect_group.get_members():
            # TODO: add timeout
            future.append((
                member,
                rpc.rpc_async(str(member),
                              RoleDispatcherElection._rpc_respond_roles)
            ))
        for member, fut in future:
            try:
                member_roles = fut.wait()
                assigned_roles.update(member_roles)
                if len(member_roles) == 0:
                    unassigned_members.append(member)
                available_members.append(member)
            except RuntimeError:
                # timeout, peer reset (disconnect) etc.
                continue

        # Assign unassigned roles.
        unassigned_roles = list(self.all_roles - assigned_roles)
        assign_map = self._plan_assign_roles(unassigned_roles,
                                             unassigned_members,
                                             available_members)

        if assign_map is None:
            # Failed to assign, restart assign process.
            return False

        future.clear()
        success = []

        # Dispatch unassigned roles to all processes.
        for member, roles in assign_map.items():
            # TODO: add timeout
            future.append(
                rpc.rpc_async(
                    str(member),
                    RoleDispatcherElection._rpc_dispatch_roles,
                    args=(roles,)
                )
            )
        for fut in future:
            try:
                fut.wait()
                success.append(True)
            except RuntimeError:
                # timeout, peer reset (disconnect) etc.
                # Failed to assign, restart assign process.
                success.append(False)
        return all(success)

    @staticmethod
    def _plan_assign_roles(roles, unassigned_members, available_members):
        # Decide how to assign unassigned roles.
        shuffle(available_members)
        result = {member: [] for member in available_members}

        # assign roles to free members first
        idx = 0
        if len(available_members) == 0:
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

    def _update_ranks(self):
        future = []
        # Get assigned roles from all processes, also test their aliveness.
        for member in self.elect_group.get_members():
            # TODO: add timeout
            future.append((
                str(member),
                rpc.rpc_async(member, RoleDispatcherElection._rpc_respond_roles)
            ))

        for member, fut in future:
            try:
                member_roles = fut.wait()
                for member_role in member_roles:
                    self.ranks[member_role] = member
            except RuntimeError:
                # timeout, peer reset (disconnect) etc.
                return False

        future.clear()
        # Push role-rank route table to all processes.
        for member in self.elect_group.get_members():
            # TODO: add timeout
            future.append((
                str(member),
                rpc.rpc_async(member, RoleDispatcherElection._rpc_set_ranks,
                              args=(self.ranks,))
            ))
        for member, fut in future:
            try:
                fut.wait()
            except RuntimeError:
                # timeout, peer reset (disconnect) etc.
                return False
        return True

    def _rpc_respond_roles(self):
        return self.roles

    def _rpc_dispatch_roles(self, roles):
        self.roles += roles
        self.role_update_cond.notify_all()

    def _rpc_set_ranks(self, new_ranks):
        self.ranks = new_ranks
