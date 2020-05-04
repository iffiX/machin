import random
import inspect
import torch
import torch.nn as nn

from .base import TorchFramework
from ..models.base import NeuralNetworkModule, NeuralNetworkWrapper
from typing import Union, Dict

from ..noise.action_space_noise import *


def soft_update(target_net: nn.Module,
                source_net: nn.Module,
                update_rate):
    """
    Soft update target network's parameters.

    Args:
        target_net: Target network to be updated.
        source_net: Source network providing new parameters.
        update_rate: Update rate.

    Returns:
        None
    """
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(),
                                       source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - update_rate) + param.data.to(target_param.device) * update_rate
            )


def hard_update(target_net: nn.Module,
                source_net: nn.Module):
    """
    Hard update (directly copy) target network's parameters.

    Args:
        target_net: Target network to be updated.
        source_net: Source network providing new parameters.
    Returns:
        None
    """

    for target_buffer, buffer in zip(target_net.buffers(),
                                     source_net.buffers()):
        target_buffer.data.copy_(buffer.data)
    for target_param, param in zip(target_net.parameters(),
                                   source_net.parameters()):
        target_param.data.copy_(param.data)


def safe_call(model, *named_args, required_argument=()):
    """
    Call a model and discard unnecessary arguments.

    Any input tensor in named_args must not be contained inside any container,
    such as list, dict, tuple, etc. Because they will be automatically moved
    to the input device of the specified model.
    """
    input_device = model.input_device
    args = inspect.getfullargspec(model.forward).args
    args_dict = {}
    if any(arg not in args for arg in required_argument):
        missing = []
        for arg in required_argument:
            if arg not in args:
                missing.append(arg)
        raise RuntimeError("Model missing required argument field(s): {}, "
                           "check your store_observe() function.".format(missing))
    for na in named_args:
        for k, v in na.items():
            if k in args:
                if t.is_tensor(v):
                    args_dict[k] = v.to(input_device)
                else:
                    args_dict[k] = v
    return model(**args_dict)


class Transition:
    def __init__(self,
                 state: Dict[str, torch.Tensor],
                 action: Dict[str, torch.Tensor],
                 next_state: Dict[str, torch.Tensor],
                 reward: Union[float, torch.Tensor],
                 terminal: bool,
                 **kwargs):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.terminal = terminal
        self.others = kwargs
        self._check_input(self)

    def __len__(self):
        return 5 + len(list(self.others.keys()))

    def __getitem__(self, item):
        return getattr(self, item)

    def to(self, device):
        for k, v in self.state.items():
            self.state[k] = v.to(device)
        for k, v in self.action.items():
            self.action[k] = v.to(device)
        for k, v in self.next_state.items():
            self.next_state[k] = v.to(device)
        if t.is_tensor(self.reward):
            self.reward = self.reward.to(device)
        return self

    @staticmethod
    def _check_input(trans):
        if any([not torch.is_tensor(t) for t in trans.state.values()]) \
                or any([not torch.is_tensor(t) for t in trans.action.values()]) \
                or any([not torch.is_tensor(t) for t in trans.next_state.values()]):
            raise RuntimeError("State, action and next_state must be dictionaries of tensors.")
        tensor_shapes = [t.shape for t in trans.state.values()] + \
                        [t.shape for t in trans.action.values()] + \
                        [t.shape for t in trans.next_state.values()]
        if isinstance(trans.reward, float):
            batch_size = 1
        elif len(trans.reward.shape) == 2 and torch.is_tensor(trans.reward):
            batch_size = trans.reward.shape[0]
        else:
            raise RuntimeError("Reward type must be a float value or a tensor of shape [batch_size, *]")
        if not all([s[0] == batch_size for s in tensor_shapes]):
            raise RuntimeError("Batch size of tensors in the transition object doesn't match")


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_device="cpu"):
        """
        Create a replay buffer instance
        Replay buffer stores a series of transition objects and functions
        as a ring buffer. The value of "state", "action", and "next_state"
        key must be a dictionary of tensors, the key of these tensors will
        be passed to your actor network and critics network as keyword
        arguments. You may store any additional info you need in the
        transition object, Values of "reward" and other keys
        will be passed to the reward function in DDPG.

        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in **kwargs will not be concatenated.

        Note:
            You should not store any tensor inside **kwargs as they will not be
            moved to the sample output device.

        Args:
            buffer_size: Maximum buffer size
            buffer_device: Device where buffer is stored
        """
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.buffer = []
        self.index = 0

    def append(self, transition: Union[Transition, Dict]):
        """
        Store a transition object to buffer.

        Args:
            transition: A transition object.
        Returns:
            None
        """
        if isinstance(transition, dict):
            transition = Transition(**transition)
        transition.to(self.buffer_device)

        if self.size() != 0 and len(self.buffer[0]) != len(transition):
            raise ValueError("Transition object length is not equal to objects stored by buffer!")
        if self.size() > self.buffer_size:
            # trim buffer to buffer_size
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        if self.size() == self.buffer_size:
            self.buffer[self.index] = transition
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(transition)

    def size(self):
        """
        Returns:
            Length of current buffer.
        """
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def sample_batch(self, batch_size, concatenate=True, device=None, sample_keys=None):
        """
        Sample a random batch from replay buffer.

        Args:
            batch_size: Maximum size of the sample.
            device:     Device to copy to.
            sample_keys: If sample_keys is specified, then only specified keys
                         of the transition object will be sampled and stacked.

        Returns:
            None if no batch is sampled.

            Or a tuple of sampled results, the tensors in "state", "action" and
            "next_state" dictionaries, along with "reward", will be concatenated
            in dimension 0. If singular reward is float, it will be turned into
            a (1, 1) tensor, then concatenated. Any other custom keys will not be
            concatenated, just put together as lists.
        """
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
            real_num = self.size()
        else:
            batch = random.sample(self.buffer, batch_size)
            real_num = batch_size

        if len(batch) == 0:
            return 0, ()

        if device is None:
            device = self.buffer_device
        if sample_keys is None:
            sample_keys = batch[0].keys()

        result = []

        for k in sample_keys:
            if k in ("state", "action", "next_state"):
                tmp_dict = {}
                for sub_k in batch[0][k].keys():
                    if concatenate:
                        tmp_dict[sub_k] = torch.cat([item[k][sub_k].to(device) for item in batch], dim=0)
                    else:
                        tmp_dict[sub_k] = [item[k][sub_k].to(device) for item in batch]
                result.append(tmp_dict)
            elif k == "reward":
                if torch.is_tensor(batch[0][k]) and len(batch[0][k].shape) > 0:
                    result.append(torch.cat([item[k].to(device) for item in batch], dim=0).view(real_num, -1))
                else:
                    result.append(torch.tensor([float(item[k]) for item in batch], device=device).view(real_num, -1))
            elif k == "terminal":
                result.append(torch.tensor([float(item[k]) for item in batch], device=device).view(real_num, -1))
            elif k == "*":
                # select custom keys
                for remain_k in batch[0].others.keys():
                    if remain_k not in ("state", "action", "next_state", "reward", "terminal"):
                        result.append([item.others[remain_k] for item in batch])
            else:
                result.append([item[k] for item in batch])
        return real_num, tuple(result)


class DDPG(TorchFramework):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, NeuralNetworkWrapper],
                 actor_target: Union[NeuralNetworkModule, NeuralNetworkWrapper],
                 critic: Union[NeuralNetworkModule, NeuralNetworkWrapper],
                 critic_target: Union[NeuralNetworkModule, NeuralNetworkWrapper],
                 optimizer,
                 criterion,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=1,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu",
                 reward_func=None,
                 action_trans_func=None):
        """
        Initialize DDPG framework.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.rpb = ReplayBuffer(replay_size, replay_device)

        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optim = optimizer(self.actor.parameters(), learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), learning_rate)

        # Make sure target and online networks have the same weight
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        self.reward_func = DDPG.bellman_function if reward_func is None else reward_func
        self.action_trans_func = DDPG.action_transform_function if action_trans_func is None else action_trans_func

        super(DDPG, self).__init__()
        self.set_top(["actor", "critic", "actor_target", "critic_target"])
        self.set_restorable(["actor_target", "critic_target"])

    def act(self, state, use_target=False):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Policy produced by actor.
        """
        if use_target:
            return safe_call(self.actor_target, state)
        else:
            return safe_call(self.actor, state)

    def act_with_noise(self, state, noise_param=(0.0, 1.0), ratio=1.0, mode="uniform", use_target=False):
        """
        Use actor network to give a policy (with uniform noise added) to the current state.

        Returns:
            Policy (with uniform noise) produced by actor.
        """
        if mode == "uniform":
            return add_uniform_noise_to_action(self.act(state, use_target), noise_param, ratio)
        elif mode == "normal":
            return add_normal_noise_to_action(self.act(state, use_target), noise_param, ratio)
        else:
            raise RuntimeError("Unknown noise type: " + str(mode))

    def criticize(self, state, action, use_target=False):
        """
        Use critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        """
        if use_target:
            return safe_call(self.critic_target, state, action)
        else:
            return safe_call(self.critic, state, action)

    def store_observe(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer. Transition samples will be used in update()
        observe() is used during training.
        """
        self.rpb.append(transition)

    def set_reward_func(self, rf):
        """
        Set reward function, default reward function is bellman function with no extra inputs
        """
        self.reward_func = rf

    def set_action_transform_function(self, tf):
        """
        Set action transform function. The transform function is used to transform the output
        of actor to the input of critic

        Action transform function must accept:
            1. raw action from the actor model
            2. concatenated next_state dictionary from the transition object
            3. any other concatenated lists of custom keys from the transition object
        and returns:
            1. a dictionary similar to action dictinary from the transition object
        """
        self.action_trans_func = tf

    def set_update_rate(self, rate=0.01):
        self.update_rate = rate

    def get_replay_buffer(self):
        return self.rpb

    def update(self, update_value=True, update_policy=True, update_targets=True, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)
        """
        batch_size, (state, action, reward, next_state, terminal, *others) = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal", "*"])

        # Update critic network first
        # Generate value reference :math: `y_i` using target actor and target critic
        with torch.no_grad():
            next_action = self.action_trans_func(self.act(next_state, True), next_state, *others)
            next_value = self.criticize(next_state, next_action, True)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal, *others)

        cur_value = self.criticize(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

        # Update actor network
        cur_action = self.action_trans_func(self.act(state), state, *others)
        act_value = self.criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            self.actor_optim.step()

        # Update target networks
        if update_targets:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)

        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        super(DDPG, self).load(model_dir, network_map, version)
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)

    def save(self, model_dir, network_map=None, version=0):
        super(DDPG, self).save(model_dir, network_map, version)

    @staticmethod
    def action_transform_function(raw_output_action, *_):
        return {"action": raw_output_action}

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * (1 - terminal) * next_value
