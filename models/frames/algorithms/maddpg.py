import copy
import itertools

import numpy as np

from .ddpg import *
from typing import Iterable

from utils.parallel import ThreadPool, cpu_count
from utils.parallel.assigner import ModelAssigner


def average_parameters(target_param: torch.Tensor, *params: Iterable[torch.Tensor]):
    target_param.data.copy_(
        torch.mean(torch.stack([p.to(target_param.device) for p in params], dim=0), dim=0)
    )
    for p in params:
        p.data.copy_(target_param)


class MADDPG(TorchFramework):
    def __init__(self,
                 agent_num,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 available_devices: Union[list, None]=None,
                 sub_policy_num=1,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=100000,
                 replay_device="cpu",
                 reward_func=None,
                 thread_num=-1):
        """
        Initialize MADDPG framework.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = ReplayBuffer(replay_size, replay_device)
        self.agent_num = agent_num

        self.actors = [copy.deepcopy(actor) for _ in range(sub_policy_num)]
        self.actor_targets = [copy.deepcopy(actor_target) for _ in range(sub_policy_num)]
        self.critics = [copy.deepcopy(critic) for _ in range(sub_policy_num)]
        self.critic_targets = [copy.deepcopy(critic_target) for _ in range(sub_policy_num)]
        self.actor_optims = [optimizer(ac.parameters(), lr=learning_rate) for ac in self.actors]
        self.critic_optims = [optimizer(cr.parameters(), lr=learning_rate) for cr in self.critics]
        self.sub_policy_num = sub_policy_num

        if available_devices is not None and len(available_devices) > 0:
            nets = self.actors + self.actor_targets + self.critics + self.critic_targets
            # only actors and critics are related
            connections = {(i, i + self.sub_policy_num * 2): 1 for i in range(self.sub_policy_num)}
            assigner = ModelAssigner(nets, connections, available_devices)
            act_asgn, actt_asgn, crt_asgn, crtt_asgn = np.array_split(assigner.assignment, 4)
            print("Actors assigned to:")
            print(act_asgn)
            print("Actors (target) assigned to:")
            print(actt_asgn)
            print("Critics assigned to:")
            print(crt_asgn)
            print("Critics (target) assigned to:")
            print(crtt_asgn)

        # create wrapper for target actors and target critics
        self.actor_target = nn.Module()
        self.critic_target = nn.Module()
        for actor_t, idx in zip(self.actor_targets, range(self.sub_policy_num)):
            self.actor_target.add_module("actor_{}".format(idx), actor_t)

        for critic_t, idx in zip(self.critic_targets, range(self.sub_policy_num)):
            self.critic_target.add_module("critic_{}".format(idx), critic_t)

        # enable multi-threading
        if not thread_num > 0:
            thread_num = cpu_count()
        self.pool = ThreadPool(thread_num)

        # Make sure target and online networks have the same weight
        with torch.no_grad():
            self.pool.starmap(hard_update, zip(self.actors, self.actor_targets))
            self.pool.starmap(hard_update, zip(self.critics, self.critic_targets))

        if lr_scheduler is not None:
            self.actor_lr_schs = [lr_scheduler(ac_opt, *lr_scheduler_params[0]) for ac_opt in self.actor_optims]
            self.critic_lr_schs = [lr_scheduler(cr_opt, *lr_scheduler_params[1]) for cr_opt in self.critic_optims]

        self.criterion = criterion

        self.reward_func = MADDPG.bellman_function if reward_func is None else reward_func

        super(MADDPG, self).__init__()
        self.set_top([])
        self.set_restorable(["actor_target", "critic_target"])

    def act(self, state, use_target=False, index=-1):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Policy produced by actor.
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            return safe_call(self.actor_targets[index], state)
        else:
            return safe_call(self.actors[index], state)

    def act_with_noise(self, state, noise_param=(0.0, 1.0),
                       ratio=1.0, mode="uniform", use_target=False, index=-1):
        """
        Use actor network to give a policy (with noise added) to the current state.

        Args:
            noise_param: A single tuple or a list of tuples specifying noise params
            for each column (last dimension) of action.
        Returns:
            Policy (with noise) produced by actor.
        """
        if mode == "uniform":
            return add_uniform_noise_to_action(self.act(state, use_target, index),
                                               noise_param, ratio)
        elif mode == "normal":
            return add_normal_noise_to_action(self.act(state, use_target, index),
                                              noise_param, ratio)
        else:
            raise RuntimeError("Unknown noise type: " + str(mode))

    def act_discreet(self, state, use_target=False, index=-1):
        """
        Use actor network to give a discreet policy to the current state.

        Note: actor network must output a probability tensor (softmax).

        Returns:
            Policy produced by actor.
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            result = safe_call(self.actor_targets[index], state)
        else:
            result = safe_call(self.actors[index], state)

        assert_output_is_probs(result)
        batch_size = result.shape[0]
        result = t.argmax(result, dim=1).view(batch_size, 1)
        return result

    def act_discreet_with_noise(self, state, use_target=False, index=-1):
        """
        Use actor network to give a policy (with noise added) to the current state.

        Note: actor network must output a probability tensor (softmax).

        Returns:
            Policy (with noise) produced by actor.
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            result = safe_call(self.actor_targets[index], state)
        else:
            result = safe_call(self.actors[index], state)

        assert_output_is_probs(result)
        dist = Categorical(result)
        batch_size = result.shape[0]
        return dist.sample([batch_size, 1])

    def criticize(self, state, all_actions, use_target=False, index=-1):
        """
        Use the first critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        Notes:
            State and action will be concatenated in dimension 1
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            return safe_call(self.critic_targets[index], state, all_actions,
                             required_argument=("all_states", "all_actions"))
        else:
            return safe_call(self.critics[index], state, all_actions,
                             required_argument=("all_states", "all_actions"))

    def store_observe(self, transition):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.

        Args:
            transition: A transition object. Could be tuple or list
        """
        self.rpb.append(transition)

    def set_reward_func(self, rf):
        """
        Set reward function, default reward function is bellman function with no extra inputs
        """
        self.reward_func = rf

    def set_update_rate(self, rate=0.01):
        self.update_rate = rate

    def get_replay_buffer(self):
        return self.rpb

    def update(self, update_value=True, update_policy=True, update_targets=True,
               average_target_parametrs=False):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)

        Note: currently agents share the same replay buffer
        """
        batch_size, (state, action, reward, next_state, terminal, agent_indexes, *others) = \
            self.rpb.sample_batch(self.batch_size,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal",
                                               "index", "*"])

        with torch.no_grad():
            # each train step will randomly select a target network to act
            all_next_actions_t = self.act(
                {"state": torch.flatten(next_state["all_states"], 0, 1)}, True)
            all_next_actions_t = {"all_actions": all_next_actions_t.view(batch_size, self.agent_num, -1)}

        def update_inner(i):
            # Update critic network first
            # Generate value reference :math: `y_i` using target actor and target critic

            with torch.no_grad():
                next_value = self.criticize(next_state, all_next_actions_t, True, i)
                next_value = next_value.view(batch_size, -1)
                y_i = self.reward_func(reward, self.discount, next_value, terminal, others)

            # action contain actions of all agents, same for state
            cur_value = self.criticize(state, action, index=i)
            value_loss = self.criterion(cur_value, y_i.to(cur_value.device))

            if update_value:
                self.critics[i].zero_grad()
                value_loss.backward()
                self.critic_optims[i].step()

            # Update actor network
            cur_all_actions = action["all_actions"].clone().detach()
            cur_all_actions[range(batch_size), agent_indexes] = \
                self.act(state, index=i).to(cur_all_actions.device)
            cur_all_actions = {"all_actions": cur_all_actions}
            act_value = self.criticize(state, cur_all_actions, index=i)

            # "-" is applied because we want to maximize J_b(u),
            # but optimizer workers by minimizing the target
            act_policy_loss = -act_value.mean()

            if update_policy:
                self.actors[i].zero_grad()
                act_policy_loss.backward()
                self.actor_optims[i].step()

            # Update target networks
            if update_targets:
                soft_update(self.actor_targets[i], self.actors[i], self.update_rate)
                soft_update(self.critic_targets[i], self.critics[i], self.update_rate)

            return act_policy_loss.item(), value_loss.item()

        all_loss = self.pool.map(update_inner, range(self.sub_policy_num))
        mean_loss = t.tensor(all_loss).mean(dim=0)

        if average_target_parametrs:
            self.average_target_parameters()

        # returns action value and policy loss
        return -mean_loss[0].item(), mean_loss[1].item()

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_schs"):
            for actor_lr_sch in self.actor_lr_schs:
                actor_lr_sch.step()
        if hasattr(self, "critic_lr_schs"):
            for critic_lr_sch in self.critic_lr_schs:
                critic_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        super(MADDPG, self).load(model_dir, network_map, version)
        with torch.no_grad():
            self.pool.starmap(hard_update, zip(self.actors, self.actor_targets))
            self.pool.starmap(hard_update, zip(self.critics, self.critic_targets))

    def save(self, model_dir, network_map=None, version=0):
        super(MADDPG, self).save(model_dir, network_map, version)

    def average_target_parameters(self):
        with torch.no_grad():
            actor_params = [net.parameters() for net in self.actor_targets]
            critic_params = [net.parameters() for net in self.critic_targets]
            self.pool.starmap(average_parameters, itertools.chain(zip(*actor_params), zip(*critic_params)))

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * (1 - terminal) * next_value
