import torch
import warnings

from utils.prep import prep_load_model

class TorchFramework:
    def __init__(self):
        self._is_top = []
        self._is_restorable = []

    def set_top(self, top):
        """
        Set top level modules.
        """
        self._is_top = top
        return self

    def set_restorable(self, restorable):
        """
        Set restorable (load & save) modules.
        """
        self._is_restorable = restorable
        return self

    def get_restorable(self):
        return self._is_restorable

    def load(self, model_dir, network_map=None, version=-1):
        """
        Load weights into modules.

        Args:
            model_dir: Save directory.
            network_map: Key is module name, value is saved name.

        Note:
            An example of network map:
            {"actor": "actor", "critic": "critic"}
        """
        network_map = {} if network_map is None else network_map
        restore_map = {}
        for r in self._is_restorable:
            if r in network_map:
                restore_map[network_map[r]] = getattr(self, r)
            else:
                warnings.warn("Load path for module \"{}\" is not specified, module name is used.".format(r),
                              RuntimeWarning)
                restore_map[r] = getattr(self, r)
        prep_load_model(model_dir, restore_map, version)

    def save(self, model_dir, network_map=None, version=-1):
        """
        Save module weights.

        Args:
            model_dir: Save directory.
            network_map: Key is module name, value is saved name.
        """
        network_map = {} if network_map is None else network_map
        if version == -1:
            version = "default"
            warnings.warn("You are using the default version to save, use custom version instead.",
                          RuntimeWarning)
        for r in self._is_restorable:
            if r in network_map:
                torch.save(getattr(self, r).state_dict(),
                           model_dir + "{}_{}.pt".format(network_map[r], version))
            else:
                warnings.warn("Save name for module \"{}\" is not specified, module name is used.".format(r),
                              RuntimeWarning)
                torch.save(getattr(self, r).state_dict(),
                           model_dir + "/{}_{}.pt".format(r, version))

    def eval(self):
        for t in self._is_top:
            getattr(self, t).eval()

    def train(self):
        for t in self._is_restorable:
            getattr(self, t).train()

    def to(self, device):
        """
        Move all modules to specified device.

        Args:
            device: torch.device class.
        """
        for t in self._is_top:
            getattr(self, t).to(device)