from machin.parallel.assigner import (
    ModelSizeEstimator,
    ModelAssigner
)
from machin.utils.helper_classes import Object

import mock
import pytest
import torch as t
import torch.nn as nn


class TestModelSizeEstimator(object):
    def test_estimator(self):
        """
        Estimate the test estimator.

        Args:
            self: (todo): write your description
        """
        model = nn.Module()
        model.register_parameter("param1",
                                 nn.Parameter(t.zeros([10], dtype=t.int8),
                                              requires_grad=False))
        model.register_parameter("param2",
                                 nn.Parameter(t.zeros([10], dtype=t.int16),
                                              requires_grad=False))
        model.register_parameter("param3",
                                 nn.Parameter(t.zeros([10], dtype=t.int32),
                                              requires_grad=False))
        model.register_parameter("param4",
                                 nn.Parameter(t.zeros([10], dtype=t.double),
                                              requires_grad=False))
        model.register_buffer("buffer1", t.zeros([10], dtype=t.float))
        assert (ModelSizeEstimator(model, size_multiplier=1)
                .get_buffer_sizes() == (10 * 4) / (1024 ** 2))
        assert (ModelSizeEstimator(model, size_multiplier=1)
                .get_parameter_sizes() ==
                (10 * 1 + 10 * 2 + 10 * 4 + 10 * 8) / (1024 ** 2))
        assert (ModelSizeEstimator(model, size_multiplier=1)
                .estimate_size() == (
                        10 * 1 + 10 * 2 + 10 * 4 + 10 * 8 + 10 * 4) / (
                        1024 ** 2))


class TestModelAssigner(object):
    # unit of size is MB
    virtual_gpus = []
    virtual_cpu = 0

    def patch_gputil_get_available(self, order):
        """
        Patch the gputililililililil.

        Args:
            self: (todo): write your description
            order: (todo): write your description
        """
        return list(range(len(self.virtual_gpus)))

    def patch_gputil_get_gpus(self):
        """
        Patch gputilus. gpus.

        Args:
            self: (todo): write your description
        """
        return [Object({"memoryFree": gsize, "load": 0})
                for gsize in self.virtual_gpus]

    def patch_psutil_virtual_memory(self):
        """
        Patch psutil psutil memory memory.

        Args:
            self: (todo): write your description
        """
        return Object({"available": self.virtual_cpu * 1024 ** 2})

    @staticmethod
    def patch_model_size_estimator(mocked_model, multiplier):
        """
        Patch size estimator for estimator.

        Args:
            mocked_model: (todo): write your description
            multiplier: (float): write your description
        """
        return Object({"estimate_size":
                       lambda: mocked_model.size * multiplier})

    @pytest.mark.parametrize("params,gpus,cpu,assignment,exception,match", [
        ({
             "models": [Object({"size": 10, "to": lambda *_: None}),
                        Object({"size": 10, "to": lambda *_: None})],
             "model_connection": {(0, 1): 1},
             "devices": ["cuda:0", "cuda:1", "cpu"],
             "model_size_multiplier": 1,
             "max_mem_ratio": 0.7,
             "cpu_weight": 0.1,
             "connection_weight": 1,
             "size_match_weight": 1e-2,
             "complexity_match_weight": 10,
             "entropy_weight": 1,
             "iterations": 500,
             "update_rate": 0.01,
             "gpu_gpu_distance": 1,
             "cpu_gpu_distance": 10,
             "move_models": True
         }, [1000, 1000], 1000, [["cuda:0", "cuda:1"],
                                 ["cuda:1", "cuda:0"]], None, None),
        ({
             "models": [Object({"size": 10, "to": lambda *_: None}),
                        Object({"size": 10, "to": lambda *_: None})],
             "model_connection": {(0, 1): 1},
             "devices": ["cuda:0", "cuda:1", "cpu"],
             "model_size_multiplier": 1,
             "max_mem_ratio": 0.7,
             "cpu_weight": 0.1,
             "connection_weight": 10,
             "size_match_weight": 1e-2,
             "complexity_match_weight": 1,
             "entropy_weight": 1,
             "iterations": 500,
             "update_rate": 0.01,
             "gpu_gpu_distance": 1,
             "cpu_gpu_distance": 10,
             "move_models": True
         }, [1000, 1000], 1000, [["cuda:0", "cuda:0"],
                                 ["cuda:1", "cuda:1"]], None, None),
        ({
             "models": [Object({"size": 600, "to": lambda *_: None}),
                        Object({"size": 600, "to": lambda *_: None})],
             "model_connection": {(0, 1): 1},
             "devices": ["cuda:0", "cuda:1", "cpu"],
             "model_size_multiplier": 1,
             "max_mem_ratio": 0.7,
             "cpu_weight": 0.1,
             "connection_weight": 1,
             "size_match_weight": 1e-2,
             "complexity_match_weight": 10,
             "entropy_weight": 1,
             "iterations": 500,
             "update_rate": 0.01,
             "gpu_gpu_distance": 1,
             "cpu_gpu_distance": 10,
             "move_models": True
         }, [1000, 1000], 1000, [["cuda:1", "cuda:0"],
                                 ["cuda:0", "cuda:1"]], None, None),
        ({
             "models": [Object({"size": 10, "to": lambda *_: None}),
                        Object({"size": 10, "to": lambda *_: None})],
             "model_connection": {(0, 1): 1},
             "devices": ["cuda:0", "cuda:1", "cpu"],
             "model_size_multiplier": 1,
             "max_mem_ratio": 0.7,
             "cpu_weight": 0.1,
             "connection_weight": 1,
             "size_match_weight": 1e-2,
             "complexity_match_weight": 10,
             "entropy_weight": 1,
             "iterations": 500,
             "update_rate": 0.01,
             "gpu_gpu_distance": 1,
             "cpu_gpu_distance": 10,
             "move_models": True
         }, [1000], 1000, [["cuda:0", "cuda:0"]], None, None),
        ({
             "models": [Object({"size": 10, "to": lambda *_: None}),
                        Object({"size": 10, "to": lambda *_: None})],
             "model_connection": {(0, 1): 1},
             "devices": ["cuda:0", "cuda:1", "cpu"],
             "model_size_multiplier": 1,
             "max_mem_ratio": 0.7,
             "cpu_weight": 0.1,
             "connection_weight": 1,
             "size_match_weight": 1e-2,
             "complexity_match_weight": 10,
             "entropy_weight": 1,
             "iterations": 500,
             "update_rate": 0.01,
             "gpu_gpu_distance": 1,
             "cpu_gpu_distance": 10,
             "move_models": True
         }, [], 1000, [["cpu", "cpu"]], None, None),
        ({
             "models": [Object({"size": 10, "to": lambda *_: None}),
                        Object({"size": 10, "to": lambda *_: None})],
             "model_connection": {(0, 1): 1},
             "devices": None,
             "model_size_multiplier": 1,
             "max_mem_ratio": 0.7,
             "cpu_weight": 0.1,
             "connection_weight": 1,
             "size_match_weight": 1e-2,
             "complexity_match_weight": 10,
             "entropy_weight": 1,
             "iterations": 500,
             "update_rate": 0.01,
             "gpu_gpu_distance": 1,
             "cpu_gpu_distance": 10,
             "move_models": True
         }, [], 1000, [["cpu", "cpu"]], None, None),
        ({
             "models": [Object({"size": 100, "to": lambda *_: None}),
                        Object({"size": 100, "to": lambda *_: None})],
             "model_connection": {(0, 1): 1},
             "devices": ["cuda:0", "cuda:1", "cpu"],
             "model_size_multiplier": 1,
             "max_mem_ratio": 0.7,
             "cpu_weight": 0.1,
             "connection_weight": 1,
             "size_match_weight": 1e-2,
             "complexity_match_weight": 10,
             "entropy_weight": 1,
             "iterations": 500,
             "update_rate": 0.01,
             "gpu_gpu_distance": 1,
             "cpu_gpu_distance": 10,
             "move_models": True
         }, [10], 10, None, RuntimeError, "Estimated model will use"),
    ])
    def test_assigner(self, params, gpus, cpu, assignment, exception, match):
        """
        Assign the given parameters.

        Args:
            self: (todo): write your description
            params: (dict): write your description
            gpus: (str): write your description
            cpu: (todo): write your description
            assignment: (todo): write your description
            exception: (todo): write your description
            match: (todo): write your description
        """
        t.manual_seed(0)
        self.virtual_gpus = gpus
        self.virtual_cpu = cpu

        with mock.patch("machin.parallel.assigner.GPUtil.getAvailable",
                        self.patch_gputil_get_available) as _p1, \
                mock.patch("machin.parallel.assigner.GPUtil.getGPUs",
                           self.patch_gputil_get_gpus) as _p2, \
                mock.patch("machin.parallel.assigner.psutil.virtual_memory",
                           self.patch_psutil_virtual_memory) as _p3, \
                mock.patch("machin.parallel.assigner.ModelSizeEstimator",
                           self.patch_model_size_estimator) as _p4:
            if exception is not None:
                with pytest.raises(exception, match=match):
                    ModelAssigner(**params)
            else:
                assigner = ModelAssigner(**params)
                real_assignment = [
                    str(dev) for dev in assigner.assignment
                ]
                assert real_assignment in assignment
