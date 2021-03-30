from typing import Union, Dict, List, Tuple, Any, Callable
from ..transition import (
    Transition,
    Scalar,
    TransitionStorageSmart,
    TransitionStorageBasic,
)
import torch as t
import random


class Buffer:
    def __init__(self, buffer_size, buffer_device="cpu", *_, **__):
        """
        Create a buffer instance.

        Buffer stores a series of transition objects and functions
        as a ring buffer. **It is not thread-safe**.

        See Also:
            :class:`.Transition`


        During sampling, the tensors in "state", "action" and "next_state"
        dictionaries, along with "reward", will be concatenated in dimension 0.
        any other custom keys specified in ``**kwargs`` will not be
        concatenated.

        Args:
            buffer_size: Maximum buffer size.
            buffer_device: Device where buffer is stored.
        """
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.buffer = TransitionStorageSmart(buffer_size)
        self.index = 0

    def append(
        self,
        transition: Union[Transition, Dict],
        required_attrs=("state", "action", "next_state", "reward", "terminal"),
    ):
        """
        Store a transition object to buffer.

        Args:
            transition: A transition object.
            required_attrs: Required attributes. Could be an empty tuple if
                no attribute is required.

        Raises:
            ``ValueError`` if transition object doesn't have required
            attributes in ``required_attrs`` or has different attributes
            compared to other transition objects stored in buffer.
        """
        if isinstance(transition, dict):
            transition = Transition(**transition)
        elif isinstance(transition, Transition):
            pass
        else:  # pragma: no cover
            raise RuntimeError(
                "Transition object must be a dict or an instance"
                " of the Transition class"
            )
        if not transition.has_keys(required_attrs):
            missing_keys = set(required_attrs) - set(transition.keys())
            raise ValueError(f"Transition object missing attributes: {missing_keys}")
        transition.to(self.buffer_device)

        if self.size() != 0 and self.buffer[0].keys() != transition.keys():
            raise ValueError("Transition object has different attributes!")

        return self.buffer.store(transition)

    def size(self):
        """
        Returns:
            Length of current buffer.
        """
        return len(self.buffer)

    def clear(self):
        """
        Remove all entries from the buffer
        """
        self.buffer.clear()

    @staticmethod
    def sample_method_random_unique(
        buffer: List[Transition], batch_size: int
    ) -> Tuple[int, List[Transition]]:
        """
        Sample unique random samples from buffer.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        if len(buffer) < batch_size:
            batch = random.sample(buffer, len(buffer))
            real_num = len(buffer)
        else:
            batch = random.sample(buffer, batch_size)
            real_num = batch_size
        return real_num, batch

    @staticmethod
    def sample_method_random(
        buffer: List[Transition], batch_size: int
    ) -> Tuple[int, List[Transition]]:
        """
        Sample random samples from buffer.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        indexes = [random.randint(0, len(buffer) - 1) for _ in range(batch_size)]
        batch = [buffer[i] for i in indexes]
        return batch_size, batch

    @staticmethod
    def sample_method_all(buffer: List[Transition], _) -> Tuple[int, List[Transition]]:
        """
        Sample all samples from buffer. Always return the whole buffer,
        will ignore the ``batch_size`` parameter.
        """
        return len(buffer), buffer

    def sample_batch(
        self,
        batch_size: int,
        concatenate: bool = True,
        device: Union[str, t.device] = None,
        sample_method: Union[Callable, str] = "random_unique",
        sample_attrs: List[str] = None,
        additional_concat_attrs: List[str] = None,
        *_,
        **__,
    ) -> Any:
        """
        Sample a random batch from buffer.

        See Also:
            Default sample methods are defined as static class methods.

            :meth:`.Buffer.sample_method_random_unique`

            :meth:`.Buffer.sample_method_random`

            :meth:`.Buffer.sample_method_all`

        Note:
            "Concatenation"
            means ``torch.cat([...], dim=0)`` for tensors,
            and ``torch.tensor([...]).view(batch_size, 1)`` for scalars.

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
                           ``func(list, batch_size)->(list, result_size)``
            concatenate: Whether concatenate state, action and next_state
                         in dimension 0.
                         If ``True``, for each value in dictionaries of major
                         attributes. and each value of sub attributes, returns
                         a concatenated tensor. Custom Attributes specified in
                         ``additional_concat_attrs`` will also be concatenated.
                         If ``False``, return a list of tensors.
            device:      Device to copy to.
            sample_attrs: If sample_keys is specified, then only specified keys
                         of the transition object will be sampled. You may use
                         ``"*"`` as a wildcard to collect remaining
                         **custom keys** as a ``dict``, you cannot collect major
                         and sub attributes using this.
                         Invalid sample attributes will be ignored.
            additional_concat_attrs: additional **custom keys** needed to be
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
                 ``additional_concat_attrs``, then lists, otherwise tensors.
        """
        if isinstance(sample_method, str):
            if not hasattr(self, "sample_method_" + sample_method):
                raise RuntimeError(
                    f"Cannot find specified sample method: {sample_method}"
                )
            sample_method = getattr(self, "sample_method_" + sample_method)
        batch_size, batch = sample_method(self.buffer, batch_size)

        if device is None:
            device = self.buffer_device

        return (
            batch_size,
            self.post_process_batch(
                batch, device, concatenate, sample_attrs, additional_concat_attrs
            ),
        )

    @classmethod
    def post_process_batch(
        cls,
        batch: List[Transition],
        device: Union[str, t.device],
        concatenate: bool,
        sample_attrs: List[str],
        additional_concat_attrs: List[str],
    ):
        """
        Post-process (concatenate) sampled batch.
        """
        result = []
        used_keys = []

        if len(batch) == 0:
            return None
        if sample_attrs is None:
            sample_attrs = batch[0].keys() if batch else []
        if additional_concat_attrs is None:
            additional_concat_attrs = []

        major_attr = set(batch[0].major_attr)
        sub_attr = set(batch[0].sub_attr)
        custom_attr = set(batch[0].custom_attr)
        for attr in sample_attrs:
            if attr in major_attr:
                tmp_dict = {}
                for sub_k in batch[0][attr].keys():
                    tmp_dict[sub_k] = cls.make_tensor_from_batch(
                        [item[attr][sub_k].to(device) for item in batch],
                        device,
                        concatenate,
                    )
                result.append(tmp_dict)
                used_keys.append(attr)
            elif attr in sub_attr:
                result.append(
                    cls.make_tensor_from_batch(
                        [item[attr] for item in batch], device, concatenate
                    )
                )
                used_keys.append(attr)
            elif attr == "*":
                # select custom keys
                tmp_dict = {}
                for remain_k in batch[0].keys():
                    if (
                        remain_k not in major_attr
                        and remain_k not in sub_attr
                        and remain_k not in used_keys
                    ):
                        tmp_dict[remain_k] = cls.make_tensor_from_batch(
                            [item[remain_k] for item in batch],
                            device,
                            concatenate and remain_k in additional_concat_attrs,
                        )
                result.append(tmp_dict)
            elif attr in custom_attr:
                result.append(
                    cls.make_tensor_from_batch(
                        [item[attr] for item in batch],
                        device,
                        concatenate and attr in additional_concat_attrs,
                    )
                )
                used_keys.append(attr)
        return tuple(result)

    @staticmethod
    def make_tensor_from_batch(
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

    def __reduce__(self):
        # for pickling
        return self.__class__, (self.buffer_size, self.buffer_device)
