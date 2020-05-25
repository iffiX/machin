from .base import NeuralNetworkModule, nn
from typing import Union
from utils.parallel.distributed import Group, get_cur_rank


class DistributedNeuralNetworkModule(NeuralNetworkModule):
    def __init__(self, group: Group):
        super(NeuralNetworkModule, self).__init__()
        self.group = group

    def set_input_dist_module(self, rank, name):
        self.input_module = (rank, name)

    def set_output_dist_module(self, rank, name):
        self.output_module = (rank, name)

    @property
    def input_device(self):
        return "cpu"

    @property
    def input_process_rank(self):
        return self._get_input_process_rank()

    def _get_input_process_rank(self):
        if self.input_module is None:
            raise RuntimeError("Input module not set.")
        else:
            if isinstance(self.input_module, tuple):
                return self.group.rpc_paired_class_sync(
                    self.input_module[0],
                    self._get_input_process_rank,
                    self.input_module[1]
                )
            else:
                return get_cur_rank()

    @property
    def output_device(self):
        return "cpu"

    @property
    def output_process_rank(self):
        return self._get_output_process_rank()

    def _get_output_process_rank(self):
        if self.output_module is None:
            raise RuntimeError("Output module not set.")
        else:
            if isinstance(self.output_module, tuple):
                return self.group.rpc_paired_class_sync(
                    self.output_module[0],
                    self._get_output_process_rank,
                    self.output_module[1]
                )
            else:
                return get_cur_rank()

    def __call__(self, *args, **kwargs):
        pass
