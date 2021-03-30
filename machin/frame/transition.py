from typing import Union, Dict, Iterable, Any, NewType
from itertools import chain
from copy import deepcopy
import torch as t
import numpy as np

Scalar = NewType("Scalar", Union[int, float, bool])


class TransitionBase:
    """
    Base class for all transitions
    """

    _inited = False

    def __init__(
        self,
        major_attr: Iterable[str],
        sub_attr: Iterable[str],
        custom_attr: Iterable[str],
        major_data: Iterable[Dict[str, t.Tensor]],
        sub_data: Iterable[Union[Scalar, t.Tensor]],
        custom_data: Iterable[Any],
    ):
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
        self._length = len(self._keys)
        self._batch_size = None

        for attr, data in zip(
            chain(major_attr, sub_attr, custom_attr),
            chain(major_data, sub_data, custom_data),
        ):
            object.__setattr__(self, attr, data)
        # will trigger _check_validity in __setattr__
        self._inited = True
        self._detach()

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        if key not in self._keys:
            raise RuntimeError(
                "You cannot dynamically set new attributes in" "a Transition object!"
            )
        object.__setattr__(self, key, value)
        self._check_validity()

    def __setattr__(self, key, value):
        if not self._inited:
            object.__setattr__(self, key, value)
        else:
            if key not in self._keys:
                raise RuntimeError(
                    "You cannot dynamically set new attributes in"
                    "a Transition object!"
                )
        if self._inited:
            self._check_validity()

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
            Ordered in: "major_attrs, sub_attrs, custom_attrs"
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

    def _detach(self):
        """
        Detach all tensors in major attributes and sub attributes, put
        data of all attributes in place, but do not copy them.

        Returns:
            Self.
        """
        for ma in self._major_attr:
            ma_data = getattr(self, ma)
            for k, v in ma_data.items():
                ma_data[k] = v.detach()
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if t.is_tensor(sa_data):
                object.__setattr__(self, sa, sa_data.detach())
        for ca in self._custom_attr:
            ca_data = getattr(self, ca)
            object.__setattr__(self, ca, ca_data)
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
                    raise ValueError(
                        f'Key "{k}" of transition major attribute "{ma}" '
                        "is an invalid tensor"
                    )
                if batch_size is None:
                    batch_size = v.shape[0]
                else:
                    if batch_size != v.shape[0]:
                        raise ValueError(
                            f'Key "{k}" of transition major attribute "{ma}" '
                            f"has invalid batch size {v.shape[0]}."
                        )
        for sa in self._sub_attr:
            sa_data = getattr(self, sa)
            if np.isscalar(sa_data):
                # will return true for inbuilt scalar types
                # like int, bool, float
                if batch_size != 1:
                    raise ValueError(
                        "Transition sub attribute "
                        f'"{sa}" is a scalar, but batch size is {batch_size}.'
                    )
            elif t.is_tensor(sa_data):
                if sa_data.dim() < 1:
                    raise ValueError(
                        f'Transition sub attribute "{sa}" is an invalid tensor.'
                    )
                elif sa_data.shape[0] != batch_size:
                    raise ValueError(
                        "Transition sub attribute "
                        f'"{sa}" has invalid batch size {sa_data.shape[0]}.'
                    )
            else:
                raise ValueError(
                    f'Transition sub attribute "{sa}" has invalid '
                    f"value {sa_data}, requires scalar or tensor."
                )
        object.__setattr__(self, "_batch_size", batch_size)


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

    def __init__(
        self,
        state: Dict[str, t.Tensor],
        action: Dict[str, t.Tensor],
        next_state: Dict[str, t.Tensor],
        reward: Union[float, t.Tensor],
        terminal: bool,
        **kwargs,
    ):
        """
        Args:
            state: Previous observed state.
            action: Action of agent.
            next_state: Next observed state.
            reward: Reward of agent.
            terminal: Whether environment has reached terminal state.
            **kwargs: Custom attributes. They are ordered in the alphabetic
                order (provided by ``sort()``) when you call ``keys()``.

        Note:
            You should not store any tensor inside ``**kwargs`` as they will
            not be moved to the sample output device.
        """
        custom_keys = sorted(kwargs.keys())
        assert isinstance(terminal, bool) or (
            t.is_tensor(terminal) and terminal.dtype == t.bool
        )
        super().__init__(
            major_attr=["state", "action", "next_state"],
            sub_attr=["reward", "terminal"],
            custom_attr=custom_keys,
            major_data=[state, action, next_state],
            sub_data=[reward, terminal],
            custom_data=[kwargs[k] for k in custom_keys],
        )

    def _check_validity(self):
        # fix batch size to 1
        super()._check_validity()
        if self._batch_size != 1:
            raise ValueError(
                "Batch size of the default transition "
                f"implementation must be 1, is {self._batch_size}"
            )


class TransitionStorageBasic(list):
    """
    TransitionStorageBasic is a linear, size-capped chunk of memory for
    transitions, it makes sure that every stored transition is copied,
    and isolated from the passed in transition object.
    """

    def __init__(self, max_size):
        """
        Args:
            max_size: Maximum size of the transition storage.
        """
        self.max_size = max_size
        self.index = 0
        super().__init__()

    def store(self, transition: TransitionBase) -> int:
        """
        Args:
            transition: Transition object to be stored

        Returns:
            The position where transition is inserted.
        """
        transition = deepcopy(transition)
        if len(self) == self.max_size:
            # ring buffer storage
            position = self.index
            self[self.index] = transition
        elif len(self) < self.max_size:
            # append if not full
            self.append(transition)
            position = len(self) - 1
        else:  # pragma: no cover
            raise RuntimeError()
        self.index = (position + 1) % self.max_size
        return position

    def clear(self):
        super().clear()


class TransitionStorageSmart(TransitionStorageBasic):
    """
    TransitionStorageSmart is a smarter, but (potentially) slower storage
    class for transitions, but in many cases it is as fast as the basic
    storage and halves memory usage because it only deep copies half of the
    states.

    TransitionStorageSmart will compare the major attributes of the
    current stored transition object with that of the last stored transition
    object. And set them to refer to the same tensor.

    Sub attributes and custom attributes will be direcly copied.
    """

    def __init__(self, max_size):
        # DOC INHERITED
        super().__init__(max_size)

    def store(self, transition: TransitionBase) -> int:
        # DOC INHERITED
        last_index = (self.index + self.max_size - 1) % self.max_size
        if last_index < len(self):
            last_transition = self[last_index]
            for ma in transition.major_attr:
                if ma == "state":
                    last_state = getattr(last_transition, "next_state", None)
                    state = transition[ma]
                    if last_state is not None:
                        for k, v in state.items():
                            if (
                                k not in last_state
                                or v.shape != last_state[k].shape
                                or v.dtype != last_state[k].dtype
                                or not v.equal(last_state[k])
                            ):
                                transition[ma] = deepcopy(transition[ma])
                                break
                        else:
                            transition[ma] = last_state
                else:
                    transition[ma] = deepcopy(transition[ma])
        else:
            for ma in transition.major_attr:
                transition[ma] = deepcopy(transition[ma])
        for sa in transition.sub_attr:
            transition[sa] = deepcopy(transition[sa])
        for ca in transition.custom_attr:
            transition[ca] = deepcopy(transition[ca])

        # store transition
        if len(self) == self.max_size:
            # ring buffer storage
            position = self.index
            self[self.index] = transition
        elif len(self) < self.max_size:
            # append if not full
            self.append(transition)
            position = len(self) - 1
        else:  # pragma: no cover
            raise RuntimeError()
        self.index = (position + 1) % self.max_size
        return position

    def clear(self):
        super().clear()
