from os.path import join
from typing import Dict
from torchviz import make_dot
import torch as t

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

    @classmethod
    def get_restorable(cls):
        """
        Get restorable modules.
        """
        return cls._is_restorable

    def enable_multiprocessing(self):
        """
        Enable multiprocessing for all modules.
        """
        for top in self._is_top:
            model = getattr(self, top)
            model.share_memory()

    def load(self, model_dir: str, network_map: Dict[str, str] = None,
             version: int = -1):
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
                    "Load path for module \"{}\" is not specified, "
                    "module name is used.".format(r)
                )
                restore_map[r] = getattr(self, r)
        prep_load_model(model_dir, restore_map, version)

    def save(self, model_dir: str, network_map: Dict[str, str] = None,
             version: int = 0):
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
                "use custom version instead.")
        for r in self._is_restorable:
            if r in network_map:
                t.save(getattr(self, r),
                       join(model_dir,
                            "{}_{}.pt".format(network_map[r], version)))
            else:
                default_logger.warning("Save name for module \"{}\" is not "
                                       "specified, module name is used."
                                       .format(r))
                t.save(getattr(self, r),
                       join(model_dir,
                            "{}_{}.pt".format(r, version)))

    def visualize_model(self,
                        final_tensor: t.Tensor,
                        name: str,
                        directory: str):
        if name in self._visualized:
            return
        else:
            self._visualized.add(name)
            g = make_dot(final_tensor)
            g.render(filename=name,
                     directory=directory,
                     view=False,
                     cleanup=False,
                     quiet=True)
