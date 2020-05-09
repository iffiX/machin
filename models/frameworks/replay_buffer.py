import torch
import random
from typing import Union, Dict


class Transition:
    """
    The default Transition class, if you have any other main attributes (attributes which are
    dictionaries, and their values are tensors) other than:
            state, action and next_state
    Then you should inherit this class and overload its to() and _check_input() functions. in your
    __init__ function, remember to call Transition constructor before you initialize other attributes.
    And don't forget to set _length and _keys attributes.
    """

    def __init__(self,
                 state: Dict[str, torch.Tensor],
                 action: Dict[str, torch.Tensor],
                 next_state: Dict[str, torch.Tensor],
                 reward: Union[float, torch.Tensor],
                 terminal: bool,
                 **kwargs):
        self._length = 5
        self._keys = ["state", "action", "next_state", "reward", "terminal"] + list(kwargs.keys())

        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.terminal = terminal
        for k, v in kwargs.items():
            self._length += 1
            setattr(self, k, v)
        self._check_input(self)

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        return getattr(self, item)

    def keys(self):
        return self._keys

    def to(self, device):
        for k, v in self.state.items():
            self.state[k] = v.to(device)
        for k, v in self.action.items():
            self.action[k] = v.to(device)
        for k, v in self.next_state.items():
            self.next_state[k] = v.to(device)
        if torch.is_tensor(self.reward):
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
    def __init__(self, buffer_size, buffer_device="cpu", main_attributes=None):
        """
        Create a replay buffer instance
        Replay buffer stores a series of transition objects and functions
        as a ring buffer. The value of "state", "action", and "next_state"
        key must be a dictionary of tensors, the key of these tensors will
        be passed to your actor network and critics network as keyword
        arguments. You may store any additional info you need in the
        transition object, Values of "reward" and other keys
        will be passed to the reward function in DDPG/PPO/....

        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in **kwargs will not be concatenated.

        Note:
            You should not store any tensor inside **kwargs as they will not be
            moved to the sample output device.

        Args:
            buffer_size: Maximum buffer size.
            buffer_device: Device where buffer is stored.
            main_attributes: Major attributes where you can store tensors.
        """
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.buffer = []
        self.index = 0
        self.main_attrs = {"state", "action", "next_state"} if main_attributes is None else main_attributes

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

    def sample_batch(self, batch_size, concatenate=True, device=None, sample_keys=None,
                     additional_concat_keys=None):
        """
        Sample a random batch from replay buffer.

        Args:
            batch_size: Maximum size of the sample.
            concatenate: Whether concatenate state, action and next_state in dimentsion 0.
                         If True, return a tensor with dim[0] = batch_size.
                         If False, return a list of tensors with dim[0] = 1.
            device:      Device to copy to.
            sample_keys: If sample_keys is specified, then only specified keys
                         of the transition object will be sampled.
            additional_concat_keys: additional custom keys needed to be concatenated, their
                                    value must be int, float or any numerical value, and must
                                    not be tensors.

        Returns:
            None if no batch is sampled.

            Or a tuple of sampled results, the tensors in "state", "action" and
            "next_state" dictionaries, along with "reward", will be concatenated
            in dimension 0 (if concatenate=True). If singular reward is float,
            it will be turned into a (1, 1) tensor, then concatenated. Any other
            custom keys will not be concatenated, just put together as lists.
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
        if additional_concat_keys is None:
            additional_concat_keys = []

        result = []

        for k in sample_keys:
            if k in self.main_attrs:
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
                for remain_k in batch[0].keys():
                    if remain_k not in ("state", "action", "next_state", "reward", "terminal"):
                        if remain_k in additional_concat_keys:
                            result.append(torch.tensor([item[remain_k] for item in batch], device=device)
                                          .view(real_num, -1))
                        else:
                            result.append([item[remain_k] for item in batch])
            else:
                if k in additional_concat_keys:
                    result.append(torch.tensor([item[k] for item in batch], device=device)
                                  .view(real_num, -1))
                else:
                    result.append([item[k] for item in batch])
        return real_num, tuple(result)
