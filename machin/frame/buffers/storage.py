from copy import deepcopy
from typing import List, Any
from abc import ABC, abstractmethod
from ..transition import TransitionBase


class TransitionStorageBase(ABC):
    """
    Base class for all transition storage.

    Definitions:
        1. The storage should be local and not shared among multiple accessors
           (thread, process, remote accessor, etc.).
        2. The transition objects will be deep-copied and not affected by any
           operation on the passed in transition objects.
        3. When the storage is limited in size, an old handle can be reused
           for new objects. But old handles occupied by the same episode should
           be prioritized for reuse.
        4. The storage handles must be hashable.
        5. The storage must be pickable (serializable), with its stored content
           accessible after restoration, whether the serialized data is full
           content or a credential to some remote storage does not matter.
    """

    @abstractmethod
    def store_episode(self, episode: List[TransitionBase]) -> List[Any]:
        """
        Args:
            episode: Episode to be stored. All transition objects in the episode
                are guaranteed to have a valid store handle.

        Returns:
            A list of handle corresponding to each stored transition from the episode.

        Raises:
            Raise value error if episode could not be fully stored due to storage
            capacity, etc.
        """

    @abstractmethod
    def clear(self):
        """
        Erase all entries in the storage.
        """

    @abstractmethod
    def __len__(self):
        """
        Returns:
            Number of transition objects in the storage. does not equal to
            the number of valid transition objects in the buffer, since some
            may be evicted.
        """

    @abstractmethod
    def __getitem__(self, key):
        """
        Args:
            key: The handle corresponding to the stored transition in the storage.

        Returns:
            The transition object
        """


class TransitionStorageBasic(TransitionStorageBase):
    """
    TransitionStorageBasic is a linear, size-capped in-memory ring storage for
    transitions, it makes sure that every stored transition is copied, and
    isolated from the passed in transition object.
    """

    def __init__(self, max_size: int, device):
        """
        Args:
            max_size: Maximum size of the transition storage.
            device: PyTorch device, string or `t.device`.
        """
        self.max_size = max_size
        self.device = device
        self.data = []
        self.index = 0
        super().__init__()

    def store_episode(self, episode: List[TransitionBase]) -> List[int]:
        """
        See Also:
            :meth:`.TransitionStorageBase.store_episode`

        Args:
            episode: Episode to be stored.

        Returns:
            A list of positions where transition is inserted. The position
            are in range [0, max_size - 1]
        """
        positions = []
        for transition in episode:
            transition = deepcopy(transition).to(self.device)
            if len(self) == self.max_size:
                # ring buffer storage
                position = self.index
                self.data[self.index] = transition
            elif len(self) < self.max_size:
                # append if not full
                self.data.append(transition)
                position = len(self) - 1
            else:  # pragma: no cover
                raise RuntimeError()
            self.index = (position + 1) % self.max_size
            positions.append(position)
        if len(set(positions)) != len(positions):
            raise ValueError("Failed to store episode.")
        return positions

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]
