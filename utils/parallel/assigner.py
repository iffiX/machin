import psutil
import GPUtil
import numpy as np
import torch as t
import torch.nn as nn

from typing import Union, List, Dict, Tuple


class ModelSizeEstimator:
    def __init__(self,
                 model: nn.Module,
                 with_input_size_multiplier=2):
        """
        Estimates the size of PyTorch models in memory
        for a given input
        """
        self.model = model
        self.size_multiplier = with_input_size_multiplier
        self.sizes = {}
        self.dtype_sizes = {}

    def get_parameter_sizes(self):
        """Get sizes of all parameters in `model`"""
        sizes, dtype_sizes = [], []

        for p in self.model.parameters():
            sizes.append(np.array(p.shape))
            dtype_sizes.append(self.get_dtype_in_bytes(p.dtype))
        self.sizes["param"] = sizes
        self.dtype_sizes["param"] = dtype_sizes

    def get_buffer_sizes(self):
        """Get sizes of all buffers in `model`"""
        sizes, dtype_sizes = [], []

        for b in self.model.buffers():
            sizes.append(np.array(b.shape))
            dtype_sizes.append(self.get_dtype_in_bytes(b.dtype))
        self.sizes["buffer"] = sizes
        self.dtype_sizes["buffer"] = dtype_sizes

    def estimate_size(self):
        """Estimate model size in memory in megabytes and bits"""
        self.get_parameter_sizes()
        self.get_buffer_sizes()
        total = np.sum(np.array([np.prod(s) for s in self.sizes["param"]]) *
                       np.array(self.dtype_sizes["param"])) + \
                np.sum(np.array([np.prod(s) for s in self.sizes["buffer"]]) *
                       np.array(self.dtype_sizes["buffer"]))

        total_megabytes = total / (1024 ** 2)
        return total_megabytes * self.size_multiplier

    @staticmethod
    def get_dtype_in_bytes(dtype: t.dtype):
        if dtype in (t.int8, t.uint8, t.bool):
            return 1
        elif dtype in (t.int16, t.float16, t.short, t.half):
            return 2
        elif dtype in (t.int32, t.float32, t.int, t.float):
            return 4
        elif dtype in (t.int64, t.float64, t.long, t.double, t.complex32):
            return 8
        elif dtype in (t.complex64,):
            return 16
        elif dtype in (t.complex128,):
            return 32


class ModelAssigner:
    def __init__(self,
                 models: List[nn.Module],
                 model_connection: Dict[Tuple[int, int], int],
                 devices: List[Union[t.device, str]] = None,
                 model_with_input_size_multiplier=2,
                 max_mem_ratio=0.5,
                 cpu_weight=1,
                 distance_weight=5,
                 size_balance_weight=1e-3,
                 complexity_balance_weight=5,
                 entropy_weight=1,
                 iterations=500,
                 update_rate=0.01):
        if devices is None:
            devices = [t.device(type="cuda", index=i) for i in GPUtil.getAvailable(order="load")]
        else:
            devices = [t.device(d) for d in devices]
            available_devices = [t.device(type="cuda", index=i) for i in GPUtil.getAvailable(order="load")]
            used_devices = []
            for d in devices:
                if d.type == "cuda" and d not in available_devices:
                    print("Warning: device {} not available, removed.".format(d))
                else:
                    used_devices.append(d)
            devices = used_devices

        if len(devices) == 0:
            devices = [t.device("cpu")]

        print("Using these devices: {}".format(devices))

        sizes = [ModelSizeEstimator(model, model_with_input_size_multiplier).estimate_size()
                 for model in models]
        device_size_capacity = []
        device_complexity_capacity = []

        gpus = GPUtil.getGPUs()
        for d in devices:
            if d.type == "cpu":
                device_size_capacity.append(int(psutil.virtual_memory().available / 1024 ** 2) * max_mem_ratio)
                device_complexity_capacity.append(cpu_weight)
            elif d.type == "cuda":
                device_size_capacity.append(gpus[d.index].memoryFree * max_mem_ratio)
                device_complexity_capacity.append(1 - gpus[d.index].load)

        if np.sum(np.array(sizes)) > np.sum(device_size_capacity):
            raise RuntimeError("Estimated model will use {:.2f} MB, but only have {:.2f} MB allowed memory in total."
                               .format(np.sum(np.array(sizes)), np.sum(device_size_capacity)))

        # assign model to devices
        # using heuristic and gradient decent
        device_num = len(devices)
        model_num = len(models)
        placement = t.randn([model_num, device_num], requires_grad=True)
        optimizer = t.optim.Adam([placement], lr=update_rate)
        model_size = t.tensor(sizes, dtype=t.float).view([1, model_num])
        size_capacity = t.tensor(device_size_capacity, dtype=t.float).view([1, device_num])
        model_complexity = model_size
        complexity_capacity = t.tensor(device_complexity_capacity, dtype=t.float).view([1, device_num])
        model_conn = t.zeros([model_num, model_num])
        for dir, conn in model_connection.items():
            model_conn[dir[0], dir[1]] = conn
        device_distance = t.zeros([device_num, device_num])
        for i in range(device_num):
            for j in range(i):
                if devices[i].type == "cpu" and devices[j].type == "cuda" \
                        or devices[i].type == "cuda" and devices[j].type == "cpu":
                    device_distance[i, j] = device_distance[j, i] = 10
                elif devices[i].type == "cuda" and devices[j].type == "cuda" \
                        and devices[i].index != devices[j].index:
                    device_distance[i, j] = device_distance[j, i] = 1

        # optimize
        for iter in range(iterations):
            self.optimize_placement(optimizer, placement,
                                    model_size, size_capacity,
                                    model_complexity, complexity_capacity,
                                    model_conn, device_distance,
                                    distance_weight,
                                    size_balance_weight,
                                    complexity_balance_weight,
                                    entropy_weight)
        self._assignment = [devices[d] for d in t.argmax(placement, dim=1).tolist()]
        for model, ass_device in zip(models, self._assignment):
            model.to(ass_device)

    @property
    def assignment(self):
        return self._assignment

    @staticmethod
    def optimize_placement(optimizer,
                           placement: t.Tensor,
                           model_size: t.Tensor,
                           size_capacity: t.Tensor,
                           model_complexity: t.Tensor,
                           complexity_capacity: t.Tensor,
                           model_connection: t.Tensor,
                           device_distance: t.Tensor,
                           distance_weight,
                           size_balance_weight,
                           complexity_balance_weight,
                           entropy_weight):
        """
        Suppose there are n models to place and m devices available
        Args:
            optimizer: optimizer of placement
            placement: shape [n, m]
            model_size: shape [1, n]
            size_capacity: shape [1, m]
            model_complexity: shape [1, n]
            complexity_capacity: shape [1, m]
            model_connection: shape [n, n]
            device_distance: shape [m, m]
        """
        placement = t.softmax(placement, dim=-1)
        model_num = placement.shape[0]

        model_distance = t.einsum("ij,mn,jn->im", [placement, placement, device_distance])
        model_distance[np.arange(model_num), np.arange(model_num)] = 0
        connection_cost = model_connection * model_distance

        # sum(model size) < capacity
        size_match_cost = t.relu(t.einsum("ij,jk->ik", [model_size, placement]) - size_capacity)

        # match computing power percent
        norm_model_cmplx = model_complexity / t.sum(model_complexity)
        norm_cmplx_capacity = complexity_capacity / t.sum(complexity_capacity)
        cmplx_match_cost = (t.einsum("ij,jk->ik", [norm_model_cmplx, placement]) - norm_cmplx_capacity) ** 2

        # entropy loss, prevent placement probability diffuse over devices
        x = placement * placement.log()
        x1 = t.zeros_like(placement)
        entropy_cost = -t.where(placement > 0, x, x1).sum(dim=-1)

        total_cost = t.mean(connection_cost) * distance_weight + \
                     t.mean(size_match_cost) * size_balance_weight + \
                     t.mean(cmplx_match_cost) * complexity_balance_weight + \
                     t.mean(entropy_cost) * entropy_weight

        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()
