from typing import Union, Dict, Iterable, Any, NewType
from itertools import chain
from copy import deepcopy
import torch as t
import numpy as np

Scalar = NewType("Scalar", Union[int, float, bool])


class TransitionBase(object):
    """
    Base class for all transitions
    """
    def __init__(self,
                 major_attr: Iterable[str],
                 sub_attr: Iterable[str],
                 custom_attr: Iterable[str],
                 major_data: Iterable[Dict[str, t.Tensor]],
                 sub_data: Iterable[Union[Scalar, t.Tensor]],
                 custom_data: Iterable[Any]):
        """
        Note:
            Major attributes store things like state, action, next_states, etc.
            They are usually **concatenated by their dictionary keys** during
            sampling, and passed as keyword arguments to actors, critics, etc.

            Sub attributes store things like terminal states, reward, etc.
            They are usually **concatenated directly** during sampling, and used
            in different algorithms.

            Custom attributes store not concatenatable values, usually user
            specified states, used in models or as special arguments in
            different algorithms. They will be collected together as a list
            during sampling, **no further concatenation is performed**.

        Args:
            major_attr: A list of major attribute names.
            sub_attr: A list of sub attribute names.
            custom_attr: A list of custom attribute names.
            major_data: Data of major attributes.
            sub_data: Data of sub attributes.
            custom_data: Data of custom attributes.
        """
        self._major_attr = list(major_attr)
        self._sub_attr = list(sub_attr)
        self._custom_attr = list(custom_attr)
        self._keys = self._major_attr + self._sub_attr + self._custom_attr
        self._length = self._major_attr + self._sub_attr + self._custom_attr

        for attr, data in zip(chain(major_attr, sub_attr, custom_attr),
                              chain(major_data, sub_data, custom_data)):
            object.__setattr__(self, attr, data)
        self._copy_and_detach()
        self._check_validity()

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        object.__setattr__(key, value)

    def __setattr__(self, key, value):
        object.__setattr__(key, value)

    @property
    def major_attr(self):
        return self._major_attr

    @property
    def sub_attr(self):
        return self._sub_attr

    @property
    def custom_attr(self):
        return self._custom_attr

    def keys(self):
        """
        Returns:
            All attribute names in current transition object.
        """
        return self._keys

    def items(self):
        """
        Returns:
            All attribute values in current transition object.
        """
        for k in self._keys:
            yield k, getattr(self, k)

    def has_keys(self, keys: Iterable[str]):
        """
        Args:
            keys: A list of keys

        Returns:
            A bool indicating whether current transition object
            contains all specified keys.
        """
        return all([k in self._keys for k in keys])

    def to(self, device: Union[str, t.device]):
        """
        Move current transition object to another device. will be
        a no-op if it already locates on that device.

        Args:
            device: A valid pytorch device.

        Returns:
            Self.
        """
        for ma in self._major_attr:
            ma_data = getattr(self, ma)
            for k, v in ma_data.items():
                ma_data[k] = v.to(device)
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if t.is_tensor(sa_data):
                object.__setattr__(self, sa, sa_data.to(device))
        return self

    def _copy_and_detach(self):
        """
        Copy and detach all tensors in major attributes and sub attributes,
        as well as deep-copying various custom attributes.

        Returns:
            Self.
        """
        for ma in self._major_attr:
            ma_data = getattr(self, ma)
            for k, v in ma_data.items():
                ma_data[k] = v.clone().detach()
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if t.is_tensor(sa_data):
                object.__setattr__(self, sa, sa_data.clone().detach())
        for ca in self._custom_attr:
            ca_data = getattr(self, ca)
            object.__setattr__(self, ca, deepcopy(ca_data))
        return self

    def _check_validity(self):
        """
        Check validity of current transition object, will check batch size,
        and major attributes' data, sub attributes' data.

        Raises:
            ``ValueError`` if anything is invalid.
        """
        batch_size = None
        for ma in self._major_attr:
            ma_data = getattr(self, ma)
            for k, v in ma_data.items():
                if not t.is_tensor(v) or v.dim() < 1:
                    raise ValueError("Key {} of transition major attribute "
                                     "{} is a invalid tensor".format(k, ma))
                if batch_size is None:
                    batch_size = v.shape[0]
                else:
                    if batch_size != v.shape[0]:
                        raise ValueError("Key {} of transition major attribute "
                                         "{} has invalid batch size {}."
                                         .format(k, ma, v.shape[0]))
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if np.isscalar(sa_data):
                # will return true for inbuilt scalar types
                # like int, bool, float
                if batch_size != 1:
                    raise ValueError("Transition sub attribute "
                                     "{} is a scalar, but batch size is {}."
                                     .format(sa, batch_size))
            elif t.is_tensor(sa_data):
                if sa_data.dim() < 1:
                    raise ValueError("Transition sub attribute "
                                     "{} is a invalid tensor.")
                elif sa_data.shape[0] != batch_size:
                    raise ValueError("Transition sub attribute "
                                     "{} has invalid batch size {}."
                                     .format(sa, sa_data.shape[0]))
            else:
                raise ValueError("Transition sub attribute {} has invalid "
                                 "value {}, requires scalar or tensor."
                                 .format(sa, sa_data))


class Transition(TransitionBase):
    """
    The default Transition class.

    Have three main attributes: ``state``, ``action`` and ``next_state``.

    Have two sub attributes: ``reward`` and ``terminal``.

    Store one transition step of one agent.
    """
    state = None  # type: Dict[str, t.Tensor]
    action = None  # type: Dict[str, t.Tensor]
    next_state = None  # type: Dict[str, t.Tensor]
    reward = None  # type: Union[float, t.Tensor]
    terminal = None  # type: bool

    def __init__(self,
                 state: Dict[str, t.Tensor],
                 action: Dict[str, t.Tensor],
                 next_state: Dict[str, t.Tensor],
                 reward: Union[float, t.Tensor],
                 terminal: bool,
                 **kwargs):
        """
        Args:
            state: Previous observed state.
            action: Action of agent.
            next_state: Next observed state.
            reward: Reward of agent.
            terminal: Whether environment has reached terminal state.
            **kwargs: Custom attributes.

        Note:
            You should not store any tensor inside ``**kwargs`` as they will
            not be moved to the sample output device.
        """
        super(Transition, self).__init__(
            major_attr=["state", "action", "next_state"],
            sub_attr=["reward", "terminal"],
            custom_attr=kwargs.keys(),
            major_data=[state, action, next_state],
            sub_data=[reward, terminal],
            custom_data=kwargs.values()
        )
