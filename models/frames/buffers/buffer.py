import torch as t
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
                 state: Dict[str, t.Tensor],
                 action: Dict[str, t.Tensor],
                 next_state: Dict[str, t.Tensor],
                 reward: Union[float, t.Tensor],
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
        self._copy()
        self._check_input(self)

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        return getattr(self, item)

    def keys(self):
        return self._keys

    def items(self):
        for k in self._keys:
            yield k, getattr(self, k)

    def has_keys(self, keys):
        return all([k in self._keys for k in keys])

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

    def _copy(self):
        for k, v in self.state.items():
            self.state[k] = v.clone().detach()
        for k, v in self.action.items():
            self.action[k] = v.clone().detach()
        for k, v in self.next_state.items():
            self.next_state[k] = v.clone().detach()
        if t.is_tensor(self.reward):
            self.reward = self.reward.clone().detach()
        return self

    @staticmethod
    def _check_input(trans):
        if any([not t.is_tensor(tr) for tr in trans.state.values()]) \
                or any([not t.is_tensor(tr) for tr in trans.action.values()]) \
                or any([not t.is_tensor(tr) for tr in trans.next_state.values()]):
            raise RuntimeError("State, action and next_state must be dictionaries of tensors.")
        tensor_shapes = [ts.shape for ts in trans.state.values()] + \
                        [ts.shape for ts in trans.action.values()] + \
                        [ts.shape for ts in trans.next_state.values()]
        if isinstance(trans.reward, float):
            batch_size = 1
        elif len(trans.reward.shape) == 2 and t.is_tensor(trans.reward):
            batch_size = trans.reward.shape[0]
        else:
            raise RuntimeError("Reward type must be a float value or a tensor of shape [batch_size, *]")
        if not all([s[0] == batch_size for s in tensor_shapes]):
            raise RuntimeError("Batch size of tensors in the transition object doesn't match")


class Buffer:
    def __init__(self, buffer_size, buffer_device="cpu", main_attributes=None, *_, **__):
        """
        Create a buffer instance
        Buffer stores a series of transition objects and functions
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

    def append(self, transition: Union[Transition, Dict],
               required_keys=("state", "action", "next_state", "reward", "terminal")):
        """
        Store a transition object to buffer.

        Args:
            transition: A transition object.
            required_keys: Required attributes.
        """
        if isinstance(transition, dict):
            transition = Transition(**transition)
        if not transition.has_keys(required_keys):
            missing_keys = set(required_keys) - set(transition.keys())
            raise RuntimeError("Transition object missing keys: {}".format(missing_keys))
        transition.to(self.buffer_device)

        if self.size() != 0 and len(self.buffer[0]) != len(transition):
            raise ValueError("Transition object length is not equal to objects stored by buffer!")
        if self.size() > self.buffer_size:
            # trim buffer to buffer_size
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        if self.size() == self.buffer_size:
            position = self.index
            self.buffer[self.index] = transition
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(transition)
            position = len(self.buffer) - 1
        return position

    def size(self):
        """
        Returns:
            Length of current buffer.
        """
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    @staticmethod
    def sample_method_random_unique(buffer, batch_size):
        if len(buffer) < batch_size:
            batch = random.sample(buffer, len(buffer))
            real_num = len(buffer)
        else:
            batch = random.sample(buffer, batch_size)
            real_num = batch_size
        return batch, real_num

    @staticmethod
    def sample_method_random(buffer, batch_size):
        indexes = [random.randint(0, len(buffer) - 1) for _ in range(batch_size)]
        batch = [buffer[i] for i in indexes]
        return batch, batch_size

    @staticmethod
    def sample_method_all(buffer, _):
        return buffer, len(buffer)

    def sample_batch(self, batch_size, concatenate=True, device=None,
                     sample_method="random_unique",
                     sample_keys=None,
                     additional_concat_keys=None,
                     *_, **__):
        """
        Sample a random batch from buffer.

        Args:
            batch_size: Maximum size of the sample.
            sample_method: Sample method, could be a string of "random", "random_unique",
                           "all", or a function(list, batch_size) -> result, result_size.
            concatenate: Whether concatenate state, action and next_state in dimension 0.
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
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError("Cannot find specified sample method: {}".format(sample_method))
            sample_method = getattr(self, "sample_method_" + sample_method)
        batch, batch_size = sample_method(self.buffer, batch_size)

        if device is None:
            device = self.buffer_device
        if sample_keys is None:
            sample_keys = batch[0].keys()
        if additional_concat_keys is None:
            additional_concat_keys = []

        return batch_size, self.post_process_batch(batch, device, concatenate,
                                                   sample_keys, additional_concat_keys)

    def post_process_batch(self, batch, device, concatenate,
                           sample_keys, additional_concat_keys):
        result = []
        used_keys = []
        for k in sample_keys:
            if k in self.main_attrs:
                tmp_dict = {}
                for sub_k in batch[0][k].keys():
                    tmp_dict[sub_k] = self.make_tensor_from_batch(
                        [item[k][sub_k].to(device) for item in batch],
                        device, concatenate
                    )
                result.append(tmp_dict)
                used_keys.append(k)
            elif k in ("reward", "terminal"):
                result.append(self.make_tensor_from_batch(
                    [item[k] for item in batch],
                    device, concatenate
                ))
                used_keys.append(k)
            elif k == "*":
                # select custom keys
                for remain_k in batch[0].keys():
                    if remain_k not in ("state", "action", "next_state", "reward", "terminal") \
                            and remain_k not in used_keys:
                        result.append(self.make_tensor_from_batch(
                            [item[remain_k] for item in batch],
                            device, concatenate and k in additional_concat_keys
                        ))
            else:
                result.append(self.make_tensor_from_batch(
                    [item[k] for item in batch],
                    device, concatenate and k in additional_concat_keys
                ))
                used_keys.append(k)
        return tuple(result)

    @staticmethod
    def make_tensor_from_batch(batch, device, concatenate):
        if len(batch) == 0:
            return None
        if concatenate:
            item = batch[0]
            batch_size = len(batch)
            if t.is_tensor(item):
                batch = [it.to(device) for it in batch]
                return t.cat(batch, dim=0).to(device)
            else:
                return t.tensor(batch, device=device).view(batch_size, -1)
        else:
            return batch

    def __reduce__(self):
        # for pickling
        return self.__class__, (self.buffer_size, self.buffer_device, self.main_attrs)
