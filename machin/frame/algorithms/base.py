from os.path import join
from typing import Dict, Any, Callable, Union
from torchviz import make_dot
import torch as t

from machin.utils.conf import Config
from machin.utils.prepare import prep_load_model
from machin.utils.logging import default_logger


class TorchFramework:
    """
    Base framework for all algorithms
    """

    _is_top = []
    _is_restorable = []

    def __init__(self):
        self._visualized = set()
        self._backward = t.autograd.backward
        # may be used by some distributed frameworks
        self.role = None

    @property
    def optimizers(self):
        raise NotImplementedError

    @optimizers.setter
    def optimizers(self, optimizers):
        raise NotImplementedError

    @property
    def lr_schedulers(self):
        raise NotImplementedError

    @property
    def top_models(self):
        models = []
        for m in self._is_top:
            models.append(getattr(self, m))
        return models

    @property
    def restorable_models(self):
        models = []
        for m in self._is_restorable:
            models.append(getattr(self, m))
        return models

    @property
    def backward_function(self):
        return self._backward

    @classmethod
    def get_top_model_names(cls):
        """
        Get attribute name of top level nn models.
        """
        return cls._is_top

    @classmethod
    def get_restorable_model_names(cls):
        """
        Get attribute name of restorable nn models.
        """
        return cls._is_restorable

    @classmethod
    def is_distributed(cls):
        """
        Whether this framework is a distributed framework which require
        multiple processes to run, and depends on ``torch.distributed`` or
        ``torch.distributed.rpc``
        """
        return False

    def set_backward_function(self, backward_func: Callable):
        """
        Replace the default backward function with a custom function.
        The default loss backward function is ``torch.autograd.backward``
        """
        assert callable(backward_func), "Backward function must be callable."
        self._backward = backward_func

    def enable_multiprocessing(self):
        """
        Enable multiprocessing for all modules.
        """
        for top in self._is_top:
            model = getattr(self, top)
            model.share_memory()

    def load(
        self, model_dir: str, network_map: Dict[str, str] = None, version: int = -1
    ):
        """
        Load models.

        An example of network map::

            {"restorable_model_1": "file_name_1",
             "restorable_model_2": "file_name_2"}

        Get keys by calling ``<Class name>.get_restorable()``

        Args:
            model_dir: Save directory.
            network_map: Key is module name, value is saved name.
            version: Version number of the save to be loaded.
        """
        network_map = {} if network_map is None else network_map
        restore_map = {}
        for r in self._is_restorable:
            if r in network_map:
                restore_map[network_map[r]] = getattr(self, r)
            else:
                default_logger.warning(
                    f'Load path for module "{r}" is not specified, module name is used.'
                )
                restore_map[r] = getattr(self, r)
        prep_load_model(model_dir, restore_map, version)

    def save(
        self, model_dir: str, network_map: Dict[str, str] = None, version: int = 0
    ):
        """
        Save models.

        An example of network map::

            {"restorable_model_1": "file_name_1",
             "restorable_model_2": "file_name_2"}

        Get keys by calling ``<Class name>.get_restorable()``

        Args:
            model_dir: Save directory.
            network_map: Key is module name, value is saved name.
            version: Version number of the new save.
        """
        network_map = {} if network_map is None else network_map
        if version == -1:
            version = "default"
            default_logger.warning(
                "You are using the default version to save, "
                "use custom version instead."
            )
        for r in self._is_restorable:
            if r in network_map:
                t.save(
                    getattr(self, r), join(model_dir, f"{network_map[r]}_{version}.pt"),
                )
            else:
                default_logger.warning(
                    'Save name for module "{r}" is not specified, module name is used.'
                )
                t.save(getattr(self, r), join(model_dir, f"{r}_{version}.pt"))

    def visualize_model(self, final_tensor: t.Tensor, name: str, directory: str):
        if name in self._visualized:
            return
        else:
            self._visualized.add(name)
            g = make_dot(final_tensor)
            g.render(
                filename=name,
                directory=directory,
                view=False,
                cleanup=False,
                quiet=True,
            )

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        raise NotImplementedError

    @classmethod
    def init_from_config(
        cls,
        config: Union[Dict[str, Any], Config],
        model_device: Union[str, t.device] = "cpu",
    ):
        raise NotImplementedError
