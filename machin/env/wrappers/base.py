from abc import ABC, abstractmethod
from typing import Union, List, Any


class ParallelWrapperBase(ABC):
    def __init__(self, *_, **__):
        pass

    @abstractmethod
    def reset(self, idx: Union[int, List[int], None] = None) -> Any:
        """
        Reset all environments if id is ``None``, otherwise reset the specific
        environment(s) with given index(es).

        Args:
            idx: Environment index(es) to be reset.

        Returns:
            Initial observation of all environments. Format is unspecified.
        """
        pass

    @abstractmethod
    def step(self, action, idx: Union[int, List[int], None] = None) -> Any:
        """
        Let specified environment(s) run one time step. specified environments
        must be active and have not reached terminal states before.

        Args:
            action: actions to take.
            idx: Environment index(es) to be run.

        Returns:
            New states of environments.
        """
        pass

    @abstractmethod
    def seed(self, seed: Union[int, List[int], None] = None) -> List[int]:
        """
        Set seed(s) for all environment(s).

        Args:
            seed: A single integer seed for all environments,
                or a list of integers for each environment,
                or None for default seed.

        Returns:
            New seed of each environment.
        """
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> Any:
        """
        Render all environments.
        """
        pass

    @abstractmethod
    def close(self) -> Any:
        """
        Close all environments.
        """
        pass

    @abstractmethod
    def active(self) -> List[int]:
        """
        Returns: Indexes of active environments.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Returns:
             Number of environments.
        """
        pass
