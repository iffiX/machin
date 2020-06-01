from time import sleep
from random import shuffle
from torch.distributed import rpc
from abc import ABC, abstractmethod
from threading import Thread, Condition, Lock


class RoleDispatcherBase(ABC):
    @abstractmethod
    def __init__(self, _rank, _world_size, _roles, _role_counts):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def get_roles(self):
        pass

    @abstractmethod
    def get_rank(self, role):
        pass

    @abstractmethod
    def notify_failure(self, role):
        pass


class RoleDispatcherSimple(RoleDispatcherBase):
    def __init__(self, rank, world_size, roles, role_counts):
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

        self.prefixed_role_count = {}
        count_sum = 0
        for role, role_count in zip(roles, role_counts):
            self.prefixed_role_count[role] = count_sum
            count_sum += role_count

        super(RoleDispatcherSimple, self).__init__(
            rank, world_size, roles, role_counts
        )

    def start(self):
        pass

    def get_roles(self):
        return self.roles

    def get_rank(self, role):
        return self.prefixed_role_count[role[0]] + role[1]

    def notify_failure(self, role):
        pass


class RoleDispatcherElection(RoleDispatcherBase):
    def __init__(self, rank, world_size, role_names, role_nums,
                 election_group):
        self.elect_group = election_group

        self.lead_thread = Thread(target=self._task_lead_group)
        self.deal_fail_thread = Thread(target=self._task_deal_failure)
        self.has_failure_cond = Condition(Lock())
        self.role_update_cond = Condition(Lock())
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
        self.lead_thread.start()
        self.deal_fail_thread.start()
        # Elect thread must be started before the elect group to
        # let it wait on the leader change signal.
        self.elect_group.start()

    def get_role_update_cond(self):
        return self.role_update_cond

    def get_roles(self):
        return self.roles

    def get_rank(self, role):
        return self.ranks[role]

    def notify_failure(self, role):
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

    def _respond_roles(self):
        return self.roles

    def _assign_roles(self):
        future = []
        assigned_roles = set()
        available_members = []
        unassigned_members = []
        for member in self.elect_group.get_members():
            # TODO: add timeout
            future.append((
                member,
                rpc.rpc_async(str(member),
                              RoleDispatcherElection._respond_roles)
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

        unassigned_roles = list(self.all_roles - assigned_roles)
        assign_map = self._plan_assign_roles(unassigned_roles,
                                             unassigned_members,
                                             available_members)

        if assign_map is None:
            # Failed to assign, restart assign process.
            return False

        future.clear()
        success = []
        for member, roles in assign_map.items():
            # TODO: add timeout
            future.append(
                rpc.rpc_async(
                    str(member),
                    RoleDispatcherElection._dispatch_roles,
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
                # assign roles in a round robin fashion
                result[available_members[idx]].append(role)
                idx = (idx + 1) % len(available_members)
        return result

    def _dispatch_roles(self, roles):
        self.roles += roles
        self.role_update_cond.notify_all()

    def _update_ranks(self):
        future = []

        for member in self.elect_group.get_members():
            # TODO: add timeout
            future.append((
                str(member),
                rpc.rpc_async(member, RoleDispatcherElection._respond_roles)
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
        for member in self.elect_group.get_members():
            # TODO: add timeout
            future.append((
                str(member),
                rpc.rpc_async(member, RoleDispatcherElection._set_ranks,
                              args=(self.ranks,))
            ))
        for member, fut in future:
            try:
                fut.wait()
            except RuntimeError:
                # timeout, peer reset (disconnect) etc.
                return False
        return True

    def _set_ranks(self, new_ranks):
        self.ranks = new_ranks
