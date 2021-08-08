from typing import Union, Dict, List, Tuple, Any, Callable
from ..transition import (
    TransitionBase,
    Transition,
    Scalar,
)
from .storage import TransitionStorageBase, TransitionStorageBasic
import torch as t
import random


class Buffer:
    def __init__(
        self,
        buffer_size: int = 1000000,
        buffer_device: Union[str, t.device] = "cpu",
        storage: TransitionStorageBase = None,
        **__,
    ):
        """
        Create a buffer instance.

        Buffer stores a series of transition objects and functions
        as a ring buffer. **It is not thread-safe**.

        See Also:
            :class:`.Transition`

        Args:
            buffer_size: Maximum buffer size.
            buffer_device: Device where buffer is stored.
            storage: Custom storage, not compatible with `buffer_size` and
                `buffer_device`.

        """
        self.storage = (
            TransitionStorageBasic(buffer_size, buffer_device)
            if storage is None
            else storage
        )

        self.transition_episode_number = {}  # type: Dict[Any, int]
        self.episode_transition_handles = {}  # type: Dict[int, List[Any]]
        self.episode_counter = 0

    def store_episode(
        self,
        episode: List[Union[TransitionBase, Dict]],
        required_attrs=("state", "action", "next_state", "reward", "terminal"),
    ):
        """
        Store an episode to the buffer.

        Note:
            If you pass in a dict type transition object, it will be automatically
            converted to ``Transition``, which requires attributes "state", "action"
            "next_state", "reward" and "terminal" to be present in the dict keys.

        Args:
            episode: A list of transition objects.
            required_attrs: Required attributes. Could be an empty tuple if
                no attribute is required.

        Raises:
            ``ValueError`` if episode is empty.
            ``ValueError`` if any transition object in the episode doesn't have
            required attributes in ``required_attrs``.
        """
        if len(episode) == 0:
            raise ValueError("Episode must be non-empty.")

        episode_number = self.episode_counter
        self.episode_counter += 1

        for idx, transition in enumerate(episode):
            if isinstance(transition, dict):
                transition = Transition(**transition)
            elif isinstance(transition, TransitionBase):
                pass
            else:  # pragma: no cover
                raise ValueError(
                    "Transition object must be a dict or an instance"
                    " of the Transition class."
                )
            if not transition.has_keys(required_attrs):
                missing_keys = set(required_attrs) - set(transition.keys())
                raise ValueError(
                    f"Transition object missing attributes: {missing_keys}, "
                    f"object is {transition}."
                )
            episode[idx] = transition

        # update episode version record
        handles = self.storage.store_episode(episode)
        for handle in handles:
            try:
                old_episode = self.transition_episode_number[handle]
            except (KeyError, IndexError):
                old_episode = None

            # evict old episode
            if old_episode is not None:
                for old_position in self.episode_transition_handles[old_episode]:
                    self.transition_episode_number.pop(old_position)
                self.episode_transition_handles.pop(old_episode)

            self.transition_episode_number[handle] = episode_number

        self.episode_transition_handles[episode_number] = handles

    def size(self):
        """
        Returns:
            Length of current buffer.
        """
        return len(self.storage)

    def clear(self):
        """
        Remove all entries from the buffer
        """
        self.storage.clear()

    def sample_batch(
        self,
        batch_size: int,
        concatenate: bool = True,
        device: Union[str, t.device] = "cpu",
        sample_method: Union[
            Callable[["Buffer", int], Tuple[List[Any], int]], str
        ] = "random_unique",
        sample_attrs: List[str] = None,
        additional_concat_custom_attrs: List[str] = None,
        *_,
        **__,
    ) -> Tuple[int, Union[None, tuple]]:
        """
        Sample a random batch from buffer, and perform concatenation.

        See Also:
            Default sample methods are defined as instance methods.

            :meth:`.Buffer.sample_method_random_unique`

            :meth:`.Buffer.sample_method_random`

            :meth:`.Buffer.sample_method_all`

        Note:
            "Concatenation" means ``torch.cat([list of tensors], dim=0)`` for tensors,
            and ``torch.tensor([list of scalars]).view(batch_size, 1)`` for scalars.

            By default, only major and sub attributes will be concatenated, in order to
            concatenate custom attributes, specify their names in
            `additional_concat_custom_attrs`.

        Warnings:
            Custom attributes must not contain tensors. And only scalar custom
            attributes can be concatenated, such as ``int``, ``float``,
            ``bool``.

        Args:
            batch_size: A hint size of the result sample. actual sample size
                        depends on your sample method.
            sample_method: Sample method, could be one of:
                           ``"random", "random_unique", "all"``,
                           or a function:
                           ``func(buffer, batch_size)->(list, result_size)``
            concatenate: Whether perform concatenation on major, sub and custom
                         attributes.
                         If ``True``, for each value in dictionaries of major
                         attributes. and each value of sub attributes, returns
                         a concatenated tensor. Custom Attributes specified in
                         ``additional_concat_custom_attrs`` will also be concatenated.
                         If ``False``, performs no concatenation.
            device:      Device to move tensors in the batch to.
            sample_attrs: If sample_keys is specified, then only specified keys
                         of the transition object will be sampled. You may use
                         ``"*"`` as a wildcard to collect remaining
                         **custom keys** as a ``dict``, you cannot collect major
                         and sub attributes using this.
                         Invalid sample attributes will be ignored.
            additional_concat_custom_attrs: additional **custom keys** needed to be
                         concatenated, will only work if ``concatenate`` is
                         ``True``.

        Returns:
            1. Batch size, Sampled attribute values in the same order as
               ``sample_keys``.

            2. Sampled attribute values is a tuple. Or ``None`` if sampled
               batch size is zero (E.g.: if buffer is empty or your sample
               size is 0 and you are not sampling using the "all" method).

               - For major attributes, result are dictionaries of tensors with
                 the same keys in your transition objects.

               - For sub attributes, result are tensors.

               - For custom attributes, if they are not in
                 ``additional_concat_custom_attrs``, then lists, otherwise tensors.

               - For wildcard selector, result is a dictionary containing unused custom
                 attributes, if they are not in ``additional_concat_custom_attrs``,
                 the values are lists, otherwise values are tensors.
        """
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError(
                    f"Cannot find specified sample method: {sample_method}"
                )
            sample_method = getattr(self, "sample_method_" + sample_method)
            batch_size, batch = sample_method(batch_size)
        else:
            batch_size, batch = sample_method(self, batch_size)

        return (
            batch_size,
            self.post_process_batch(
                batch, device, concatenate, sample_attrs, additional_concat_custom_attrs
            ),
        )

    def sample_method_random_unique(
        self, batch_size: int,
    ) -> Tuple[int, List[Transition]]:
        """
        Sample unique random samples from buffer.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        batch_size = min(len(self.transition_episode_number), batch_size)
        batch_handles = random.sample(
            list(self.transition_episode_number.keys()), k=batch_size
        )
        batch = [self.storage[bh] for bh in batch_handles]
        return batch_size, batch

    def sample_method_random(self, batch_size: int,) -> Tuple[int, List[Transition]]:
        """
        Sample random samples from buffer.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        batch_size = min(len(self.transition_episode_number), batch_size)
        batch_handles = random.choices(
            list(self.transition_episode_number.keys()), k=batch_size
        )
        batch = [self.storage[bh] for bh in batch_handles]
        return batch_size, batch

    def sample_method_all(self, _,) -> Tuple[int, List[Transition]]:
        """
        Sample all samples from buffer, will ignore the ``batch_size`` parameter.
        """
        batch = [self.storage[bh] for bh in self.transition_episode_number.keys()]
        return len(self.transition_episode_number), batch

    def post_process_batch(
        self,
        batch: List[Transition],
        device: Union[str, t.device],
        concatenate: bool,
        sample_attrs: List[str],
        additional_concat_custom_attrs: List[str],
    ):
        """
        Post-process sampled batch.
        """
        result = []
        used_keys = []

        if len(batch) == 0:
            return None
        if sample_attrs is None:
            sample_attrs = batch[0].keys() if batch else []
        if additional_concat_custom_attrs is None:
            additional_concat_custom_attrs = []

        major_attr = set(batch[0].major_attr)
        sub_attr = set(batch[0].sub_attr)
        custom_attr = set(batch[0].custom_attr)
        for attr in sample_attrs:
            if attr in major_attr:
                tmp_dict = {}
                for sub_k in batch[0][attr].keys():
                    tmp_dict[sub_k] = self.post_process_attribute(
                        attr,
                        sub_k,
                        self.make_tensor_from_batch(
                            self.pre_process_attribute(
                                attr,
                                sub_k,
                                [item[attr][sub_k].to(device) for item in batch],
                            ),
                            device,
                            concatenate,
                        ),
                    )
                result.append(tmp_dict)
                used_keys.append(attr)
            elif attr in sub_attr:
                result.append(
                    self.post_process_attribute(
                        attr,
                        None,
                        self.make_tensor_from_batch(
                            self.pre_process_attribute(
                                attr, None, [item[attr] for item in batch]
                            ),
                            device,
                            concatenate,
                        ),
                    )
                )
                used_keys.append(attr)
            elif attr in custom_attr:
                result.append(
                    self.post_process_attribute(
                        attr,
                        None,
                        self.make_tensor_from_batch(
                            self.pre_process_attribute(
                                attr, None, [item[attr] for item in batch]
                            ),
                            device,
                            concatenate and attr in additional_concat_custom_attrs,
                        ),
                    )
                )
                used_keys.append(attr)
            elif attr == "*":
                # select custom keys
                tmp_dict = {}
                for remain_k in custom_attr:
                    if remain_k not in used_keys:
                        tmp_dict[remain_k] = self.post_process_attribute(
                            attr,
                            None,
                            self.make_tensor_from_batch(
                                self.pre_process_attribute(
                                    attr, None, [item[remain_k] for item in batch]
                                ),
                                device,
                                concatenate
                                and remain_k in additional_concat_custom_attrs,
                            ),
                        )
                        used_keys.append(remain_k)
                result.append(tmp_dict)
        return tuple(result)

    def pre_process_attribute(
        self, attribute: Any, sub_key: Any, values: List[Union[Scalar, t.Tensor]]
    ):
        """
        Pre-process attribute items, method :meth:`.Buffer.make_tensor_from_batch`
        will use the result from this function and assumes processed attribute items
        to be one of:

        1. A list of tensors that's concatenable in dimension 0.
        2. A list of values that's transformable to a tensor.

        In case you want to implement custom padding for each item of an
        attribute, or other custom preprocess, please override this method.

        See Also:
            `This issue <https://github.com/iffiX/machin/issues/8>`_

        Args:
            attribute: Attribute key, such as "state", "next_state", etc.
            sub_key: Sub key in attribute if attribute is a major attribute,
                set to `None` if attribute is a sub attribute or a custom attribute.
            values: Sampled lists of attribute items.
        """
        return values

    def make_tensor_from_batch(
        self,
        batch: List[Union[Scalar, t.Tensor]],
        device: Union[str, t.device],
        concatenate: bool,
    ):
        """
        Make a tensor from a batch of data.
        Will concatenate input tensors in dimension 0.
        Or create a tensor of size (batch_size, 1) for scalars.

        Args:
            batch: Batch data.
            device: Device to move data to
            concatenate: Whether performing concatenation.

        Returns:
            Original batch if batch is empty,
            or tensor depends on your data (if concatenate),
            or original batch (if not concatenate).
        """
        if concatenate and len(batch) != 0:
            item = batch[0]
            batch_size = len(batch)
            if t.is_tensor(item):
                batch = [it.to(device) for it in batch]
                return t.cat(batch, dim=0).to(device)
            else:
                try:
                    return t.tensor(batch, device=device).view(batch_size, -1)
                except Exception:
                    raise ValueError(f"Batch not concatenable: {batch}")
        else:
            return batch

    def post_process_attribute(
        self,
        attribute: Any,
        sub_key: Any,
        values: Union[List[Union[Scalar, t.Tensor]], t.Tensor],
    ):
        """
        Post-process concatenated attribute items. Values are processed results from
        the method :meth:`.Buffer.make_tensor_from_batch`, either a list of not
        concatenated values, or a concatenated tensor.

        Args:
            attribute: Attribute key, such as "state", "next_state", etc.
            sub_key: Sub key in attribute if attribute is a major attribute,
                set to `None` if attribute is a sub attribute or a custom attribute.
            values: (Not) Concatenated attribute items.
        """
        return values
