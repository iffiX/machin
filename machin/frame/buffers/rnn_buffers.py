from typing import Union, Dict, List, Tuple, Any
from machin.parallel.distributed import RpcGroup
from ..transition import (
    TransitionBase,
    Transition,
    Scalar,
)
from .buffer import Buffer
from .buffer_d import DistributedBuffer
from .prioritized_buffer import PrioritizedBuffer
from .prioritized_buffer_d import DistributedPrioritizedBuffer
from .storage import TransitionStorageBase

import random
import torch as t
import numpy as np


class RNNBuffer(Buffer):
    def __init__(
        self,
        sample_length: int,
        sample_dimension: int = 1,
        buffer_size: int = 1000000,
        buffer_device: Union[str, t.device] = "cpu",
        storage: TransitionStorageBase = None,
        **kwargs,
    ):
        """
        Create a RNN buffer instance, which samples a fixed-length sequence
        from an episode.

        See Also:
            :class:`.Buffer`

        Note:
            `sample_dimension` controls the sampled sequence dimension to insert into
            the sampled batch. For example, if one step in an episode is of shape::

                [1, feature_number]

            With 1 being the batch dimension for concatenation, if
            `sample_dimension` is set to 1, then the shape of the output would be::

                [batch_size, sample_length, feature_number]

            if `sample_dimension` is set to 2, then the shape would be::

                [batch_size, feature_number, sample_length]

            `sample_dimension` **must** be a non-negative integer in range
            `[0, step_dimension_number]`.

        Note:
            When calling `RNNBuffer.sample_batch`:

            If `concatenate` is set to `False` during sampling, then the sampled
            results (eg: sub-key values of major attributes, sub attributes)
            will be `List[List[Any]]` instead of `List[Any]`, with each sampled
            sequence being the inner list.

        Args:
            sample_length: Length of the sampled sequence.
            sample_dimension: Dimension number for the sampled sequence, default is 1,
                which means after the batch dimension.
            buffer_size: Maximum buffer size.
            buffer_device: Device where buffer is stored.
            storage: Custom storage, not compatible with `buffer_size` and
                `buffer_device`.
        """
        super().__init__(
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            storage=storage,
            **kwargs,
        )
        self.sample_length = sample_length
        self.sample_dimension = sample_dimension

    def sample_method_random_unique(
        self, batch_size: int
    ) -> Tuple[int, List[Transition]]:
        """
        For each sample in batch, first sample a unique random episode,
        then sample a random starting point in the episode.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        valid_episodes = [
            it[0]
            for it in self.episode_transition_handles.items()
            if len(it[1]) >= self.sample_length
        ]
        batch_size = min(len(valid_episodes), batch_size)
        selected_episodes = random.sample(valid_episodes, k=batch_size)
        start_positions = [
            random.randint(
                0, len(self.episode_transition_handles[ep]) - self.sample_length
            )
            for ep in selected_episodes
        ]
        batch = [
            self.storage[bh]
            for ep, pos in zip(selected_episodes, start_positions)
            for bh in self.episode_transition_handles[ep][
                pos : pos + self.sample_length
            ]
        ]
        return len(selected_episodes), batch

    def sample_method_random(self, batch_size: int,) -> Tuple[int, List[Transition]]:
        """
        For each sample in batch, first sample a random episode,
        then sample a random starting point in the episode.

        Note:
            Sampled size could be any value from 0 to ``batch_size``.
        """
        valid_episodes = [
            it[0]
            for it in self.episode_transition_handles.items()
            if len(it[1]) >= self.sample_length
        ]
        batch_size = min(len(valid_episodes), batch_size)
        selected_episodes = random.choices(valid_episodes, k=batch_size)
        start_positions = [
            random.randint(
                0, len(self.episode_transition_handles[ep]) - self.sample_length
            )
            for ep in selected_episodes
        ]
        batch = [
            self.storage[bh]
            for ep, pos in zip(selected_episodes, start_positions)
            for bh in self.episode_transition_handles[ep][
                pos : pos + self.sample_length
            ]
        ]
        return len(selected_episodes), batch

    def sample_method_all(self, _) -> Tuple[int, List[Transition]]:
        """
        For each step in any episode, if starting from that step, the remaining
        steps is longer than/equal to the sample length, sample a sequence,
        otherwise ignore it.

        Will ignore the ``batch_size`` parameter.
        """
        valid_episodes = [
            it[0]
            for it in self.episode_transition_handles.items()
            if len(it[1]) >= self.sample_length
        ]
        batch = []
        batch_size = 0
        for ep in valid_episodes:
            episode_handles = self.episode_transition_handles[ep]
            for pos in range(len(episode_handles) - self.sample_length + 1):
                batch += [
                    self.storage[bh]
                    for bh in self.episode_transition_handles[ep][
                        pos : pos + self.sample_length
                    ]
                ]
                batch_size += 1

        return batch_size, batch

    def post_process_attribute(
        self,
        attribute: Any,
        sub_key: Any,
        values: Union[List[Union[Scalar, t.Tensor]], t.Tensor],
    ):
        if isinstance(values, list):
            return [
                values[i : i + self.sample_length]
                for i in range(0, len(values), self.sample_length)
            ]
        else:
            batch_size = values.shape[0]
            new_shape = [
                int(batch_size / self.sample_length),
                self.sample_length,
            ] + list(values.shape)[1:]
            return values.reshape(new_shape)


class RNNDistributedBuffer(RNNBuffer, DistributedBuffer):
    def __init__(
        self,
        buffer_name: str,
        group: RpcGroup,
        sample_length: int,
        sample_dimension: int = 1,
        buffer_size: int = 1000000,
        buffer_device: Union[str, t.device] = "cpu",
        storage: TransitionStorageBase = None,
        **kwargs,
    ):
        """
        Create a distributed RNN buffer instance, which samples a fixed-length sequence
        from an episode.

        See Also:
            :class:`.RNNBuffer`
            :class:`.DistributedBuffer`

        Note:
            `sample_dimension` controls the sampled sequence dimension to insert into
            the sampled batch. For example, if one step in an episode is of shape::

                [1, feature_number]

            With 1 being the batch dimension for concatenation, if
            `sample_dimension` is set to 1, then the shape of the output would be::

                [batch_size, sample_length, feature_number]

            if `sample_dimension` is set to 2, then the shape would be::

                [batch_size, feature_number, sample_length]

            `sample_dimension` **must** be a non-negative integer in range
            `[0, step_dimension_number]`.

        Note:
            When calling `RNNBuffer.sample_batch`:

            If `concatenate` is set to `False` during sampling, then the sampled
            results (eg: sub-key values of major attributes, sub attributes)
            will be `List[List[Any]]` instead of `List[Any]`, with each sampled
            sequence being the inner list.

        Args:
            buffer_name: A unique name of your buffer for registration in the group.
            group: Process group which holds this buffer.
            sample_length: Length of the sampled sequence.
            sample_dimension: Dimension number for the sampled sequence, default is 1,
                which means after the batch dimension.
            buffer_size: Maximum buffer size.
            buffer_device: Device where buffer is stored.
            storage: Custom storage, not compatible with `buffer_size` and
                `buffer_device`.
        """
        super().__init__(
            buffer_name=buffer_name,
            group=group,
            sample_length=sample_length,
            sample_dimension=sample_dimension,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            storage=storage,
            **kwargs,
        )


class RNNPrioritizedBuffer(RNNBuffer, PrioritizedBuffer):
    def __init__(
        self,
        sample_length: int,
        sample_dimension: int = 1,
        buffer_size: int = 1000000,
        buffer_device: Union[str, t.device] = "cpu",
        epsilon: float = 1e-2,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.001,
        **kwargs,
    ):
        """
        Create a RNN prioritized buffer instance, which samples a fixed-length
        sequence by importance weight from an episode.

        See Also:
            :class:`.RNNBuffer`
            :class:`.PrioritizedBuffer`

        Note:
            `sample_dimension` controls the sampled sequence dimension to insert into
            the sampled batch. For example, if one step in an episode is of shape::

                [1, feature_number]

            With 1 being the batch dimension for concatenation, if
            `sample_dimension` is set to 1, then the shape of the output would be::

                [batch_size, sample_length, feature_number]

            if `sample_dimension` is set to 2, then the shape would be::

                [batch_size, feature_number, sample_length]

            `sample_dimension` **must** be a non-negative integer in range
            `[0, step_dimension_number]`.

        Note:
            When calling `RNNPrioritizedBuffer.sample_batch`:

            If `concatenate` is set to `False` during sampling, then the sampled
            results (eg: sub-key values of major attributes, sub attributes)
            will be `List[List[Any]]` instead of `List[Any]`, with each sampled
            sequence being the inner list.

        Args:
            sample_length: Length of the sampled sequence.
            sample_dimension: Dimension number for the sampled sequence, default is 1,
                which means after the batch dimension.
            buffer_size: Maximum buffer size.
            buffer_device: Device where buffer is stored.
            epsilon: A small positive constant used to prevent edge-case
                zero weight transitions from never being visited.
            alpha: Prioritization weight. Used during transition sampling:
                :math:`j \\sim P(j)=p_{j}^{\\alpha} / \
                        \\sum_i p_{i}^{\\alpha}`.
                When ``alpha = 0``, all samples have the same probability
                to be sampled.
                When ``alpha = 1``, all samples are drawn uniformly according
                to their weight.
            beta: Bias correcting weight. When ``beta = 1``, bias introduced
                by prioritized replay will be corrected. Used during
                importance weight calculation:
                :math:`w_j=(N \\cdot P(j))^{-\\beta}/max_i w_i`
            beta_increment_per_sampling:
                Beta increase step size, will gradually increase ``beta`` to 1.
        """
        super().__init__(
            sample_length=sample_length,
            sample_dimension=sample_dimension,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            epsilon=epsilon,
            alpha=alpha,
            beta=beta,
            beta_increment_per_sampling=beta_increment_per_sampling,
            **kwargs,
        )

    def store_episode(
        self,
        episode: List[Union[TransitionBase, Dict]],
        priorities: Union[List[float], None] = None,
        required_attrs=("state", "action", "next_state", "reward", "terminal"),
    ):
        """
        Store an episode to the buffer.

        Args:
            episode: A list of transition objects.
            priorities: Priority of each transition in the episode.
            required_attrs: Required attributes. Could be an empty tuple if
                no attribute is required.

        Raises:
            ``ValueError`` if episode is empty.
            ``ValueError`` if any transition object in the episode doesn't have
            required attributes in ``required_attrs``.
        """
        super(PrioritizedBuffer, self).store_episode(episode, required_attrs)
        episode_number = self.episode_counter - 1
        positions = self.episode_transition_handles[episode_number]

        # force priority of steps that will not form a complete RNN sample
        # sequence to be 0, so they will never be sampled

        if priorities is None:
            # the initialization method used in the original essay
            priority = self._normalize_priority(self.wt_tree.get_leaf_max())
            priorities = [
                priority if i + self.sample_length <= len(episode) else 0
                for i in range(len(episode))
            ]
        else:
            if len(episode) < self.sample_length:
                priorities[:] = 0
            else:
                priorities = self._normalize_priority(priorities)
                priorities[len(episode) - self.sample_length + 1 :] = 0
        self.wt_tree.update_leaf_batch(priorities, positions)

    def sample_batch(
        self,
        batch_size: int,
        concatenate: bool = True,
        device: Union[str, t.device] = "cpu",
        sample_attrs: List[str] = None,
        additional_concat_custom_attrs: List[str] = None,
        *_,
        **__,
    ) -> Tuple[
        int, Union[None, tuple], Union[None, np.ndarray], Union[None, np.ndarray]
    ]:
        # DOC INHERITED

        if batch_size <= 0 or self.size() == 0:
            return 0, None, None, None

        # Since the linear ring storage used by prioritized buffers will always
        # overwrite old buffers from the start, there is no need to worry about
        # sampling incomplete sequences.

        index, is_weight = self.sample_index_and_weight(batch_size)
        batch = [
            self.storage[i]
            for idx in index
            for i in range(idx, idx + self.sample_length)
        ]
        result = self.post_process_batch(
            batch, device, concatenate, sample_attrs, additional_concat_custom_attrs
        )
        return len(index), result, index, is_weight


class RNNDistributedPrioritizedBuffer(RNNBuffer, DistributedPrioritizedBuffer):
    def __init__(
        self,
        buffer_name: str,
        group: RpcGroup,
        sample_length: int,
        sample_dimension: int = 1,
        buffer_size: int = 1000000,
        buffer_device: Union[str, t.device] = "cpu",
        epsilon: float = 1e-2,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.001,
        **kwargs,
    ):
        """
        Create a RNN distributed prioritized buffer instance, which samples a
        fixed-length sequence by importance weight from an episode.

        See Also:
            :class:`.RNNBuffer`
            :class:`.DistributedPrioritizedBuffer`

        Note:
            `sample_dimension` controls the sampled sequence dimension to insert into
            the sampled batch. For example, if one step in an episode is of shape::

                [1, feature_number]

            With 1 being the batch dimension for concatenation, if
            `sample_dimension` is set to 1, then the shape of the output would be::

                [batch_size, sample_length, feature_number]

            if `sample_dimension` is set to 2, then the shape would be::

                [batch_size, feature_number, sample_length]

            `sample_dimension` **must** be a non-negative integer in range
            `[0, step_dimension_number]`.

        Note:
            When calling `RNNDistributedPrioritizedBuffer.sample_batch`:

            If `concatenate` is set to `False` during sampling, then the sampled
            results (eg: sub-key values of major attributes, sub attributes)
            will be `List[List[Any]]` instead of `List[Any]`, with each sampled
            sequence being the inner list.

        Args:
            buffer_name: A unique name of your buffer for registration in the group.
            group: Process group which holds this buffer.
            sample_length: Length of the sampled sequence.
            sample_dimension: Dimension number for the sampled sequence, default is 1,
                which means after the batch dimension.
            buffer_size: Maximum buffer size.
            buffer_device: Device where buffer is stored.
            epsilon: A small positive constant used to prevent edge-case
                zero weight transitions from never being visited.
            alpha: Prioritization weight. Used during transition sampling:
                :math:`j \\sim P(j)=p_{j}^{\\alpha} / \
                        \\sum_i p_{i}^{\\alpha}`.
                When ``alpha = 0``, all samples have the same probability
                to be sampled.
                When ``alpha = 1``, all samples are drawn uniformly according
                to their weight.
            beta: Bias correcting weight. When ``beta = 1``, bias introduced
                by prioritized replay will be corrected. Used during
                importance weight calculation:
                :math:`w_j=(N \\cdot P(j))^{-\\beta}/max_i w_i`
            beta_increment_per_sampling:
                Beta increase step size, will gradually increase ``beta`` to 1.
        """
        super().__init__(
            buffer_name=buffer_name,
            group=group,
            sample_length=sample_length,
            sample_dimension=sample_dimension,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            epsilon=epsilon,
            alpha=alpha,
            beta=beta,
            beta_increment_per_sampling=beta_increment_per_sampling,
            **kwargs,
        )

    def store_episode(
        self,
        episode: List[Union[TransitionBase, Dict]],
        priorities: Union[List[float], None] = None,
        required_attrs=("state", "action", "next_state", "reward", "terminal"),
    ):
        # DOC INHERITED
        with self.wr_lock:
            super(DistributedPrioritizedBuffer, self).store_episode(
                episode, required_attrs
            )
            episode_number = self.episode_counter - 1
            positions = self.episode_transition_handles[episode_number]

            # force priority of steps that will not form a complete RNN sample
            # sequence to be 0, so they will never be sampled

            if priorities is None:
                # the initialization method used in the original essay
                priority = self._normalize_priority(self.wt_tree.get_leaf_max())
                priorities = [
                    priority if i + self.sample_length <= len(episode) else 0
                    for i in range(len(episode))
                ]
            else:
                if len(episode) < self.sample_length:
                    priorities[:] = 0
                else:
                    priorities = self._normalize_priority(priorities)
                    priorities[len(episode) - self.sample_length + 1 :] = 0
            self.wt_tree.update_leaf_batch(priorities, positions)

    def _sample_service(self, batch_size, all_weight_sum):  # pragma: no cover
        # the local batch size
        with self.wr_lock:
            if batch_size <= 0 or self.size() == 0:
                return 0, None, None, None

            # Since the linear ring storage used by prioritized buffers will always
            # overwrite old buffers from the start, there is no need to worry about
            # sampling incomplete sequences.

            index, is_weight = self.sample_index_and_weight(batch_size, all_weight_sum)
            version = self.buffer_version_table[index]
            batch = [
                self.storage[i]
                for idx in index
                for i in range(idx, idx + self.sample_length)
            ]
            return len(index), batch, index, version, is_weight
