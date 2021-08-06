from .buffer import *
from .buffer_d import *
from .prioritized_buffer import PrioritizedBuffer
from .prioritized_buffer_d import DistributedPrioritizedBuffer
from .storage import TransitionStorageBase

import random
import itertools
import torch as t


class RNNBuffer(Buffer):
    def __init__(
        self,
        sample_length: int,
        sample_dimension: int = 1,
        buffer_size: int = 1000000,
        buffer_device: Union[str, t.device] = "cpu",
        storage: TransitionStorageBase = None,
        *_,
        **__,
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
            When calling :meth:`RNNBuffer.sample_batch`:

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
            buffer_size=buffer_size, buffer_device=buffer_device, storage=storage
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
        *_,
        **__,
    ):
        super().__init__(
            buffer_name=buffer_name,
            group=group,
            sample_length=sample_length,
            sample_dimension=sample_dimension,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
        )
