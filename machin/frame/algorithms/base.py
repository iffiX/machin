from typing import Dict, List
from torchviz import make_dot
import torch as t

from machin.utils.prep import prep_load_model
from machin.utils.logging import default_logger


class TorchFramework:
    """
    Base framework for all algorithms
    """
    def __init__(self):
        self._is_top = []
        self._is_restorable = []
        self._visualized = set()

    def set_top(self, top):
        """
        Set top level modules.
        """
        self._is_top = top
        return self

    def set_restorable(self, restorable: List[str]):
        """
        Set restorable (loadable & savable) modules.
        """
        self._is_restorable = restorable
        return self

    def get_restorable(self):
        """
        Get restorable modules.
        """
        return self._is_restorable

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

            {"actor": "actor_file_name", "critic": "critic_file_name"}

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
                    "module name is used.".format(r),
                    RuntimeWarning
                )
                restore_map[r] = getattr(self, r)
        prep_load_model(model_dir, restore_map, version)

    def save(self, model_dir: str, network_map: Dict[str, str] = None,
             version: int = 0):
        """
        Save models.

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
                t.save(getattr(self, r).state_dict(),
                       model_dir + "{}_{}.pt".format(network_map[r], version))
            else:
                default_logger.warning("Save name for module \"{}\" is not "
                                       "specified, module name is used."
                                       .format(r))
                t.save(getattr(self, r).state_dict(),
                       model_dir + "/{}_{}.pt".format(r, version))

    def visualize_model(self, final_tensor: t.Tensor, name: str):
        if name in self._visualized:
            return
        else:
            g = make_dot(final_tensor)
            g.view(filename=name + ".gv")
