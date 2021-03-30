from typing import Union, List, Dict, Tuple
from machin.utils.logging import default_logger
import psutil
import GPUtil
import numpy as np
import torch as t
import torch.nn as nn


class ModelSizeEstimator:
    """
    Size estimator for pytorch modules.
    """

    def __init__(self, model: nn.Module, size_multiplier=2):
        """
        Estimates the size of PyTorch models in memory.

        Note:
            This estimator can only estimate the total size of parameters and
            buffers. Therefore we need to multiply the raw estimated size with
            a correction coefficient to reserve enough space for models.

        Args:
            model: Model to be estimated.
            size_multiplier: Model estimated size will be
                multiplied with this value, to ensure enough space
                will be reserved to contain your model and inputs.
        """
        self.model = model
        self.size_multiplier = size_multiplier
        self.sizes = {}
        self.dtype_sizes = {}

    def get_parameter_sizes(self) -> float:
        """Get sizes of all parameters in ``model`` in mega bytes."""
        sizes, dtype_sizes = [], []

        for param in self.model.parameters():
            sizes.append(np.array(param.shape))
            dtype_sizes.append(self._get_dtype_in_bytes(param.dtype))
        self.sizes["param"] = sizes
        self.dtype_sizes["param"] = dtype_sizes
        return float(
            np.sum(
                np.array([np.prod(s) for s in self.sizes["param"]])
                * np.array(self.dtype_sizes["param"])
            )
        ) / (1024 ** 2)

    def get_buffer_sizes(self) -> float:
        """Get sizes of all buffers in ``model`` in mega bytes."""
        sizes, dtype_sizes = [], []

        for buffer in self.model.buffers():
            sizes.append(np.array(buffer.shape))
            dtype_sizes.append(self._get_dtype_in_bytes(buffer.dtype))
        self.sizes["buffer"] = sizes
        self.dtype_sizes["buffer"] = dtype_sizes
        return float(
            np.sum(
                np.array([np.prod(s) for s in self.sizes["buffer"]])
                * np.array(self.dtype_sizes["buffer"])
            )
        ) / (1024 ** 2)

    def estimate_size(self):
        """Estimate model size in memory in megabytes."""
        total = self.get_parameter_sizes() + self.get_buffer_sizes()
        return total * self.size_multiplier

    @staticmethod
    def _get_dtype_in_bytes(dtype: t.dtype):
        if dtype in (t.int8, t.uint8, t.bool):
            return 1
        elif dtype in (t.int16, t.float16, t.short, t.half):
            return 2
        elif dtype in (t.int32, t.float32, t.int, t.float):
            return 4
        elif dtype in (t.int64, t.float64, t.long, t.double, t.complex32):
            return 8
        else:  # pragma: no cover
            raise ValueError("Invalid data type.")


class ModelAssigner:
    """
    Assigner for pytorch modules.
    """

    def __init__(
        self,
        models: List[nn.Module],
        model_connection: Dict[Tuple[int, int], int],
        devices: List[Union[t.device, str]] = None,
        model_size_multiplier=2,
        max_mem_ratio=0.5,
        cpu_weight=0,
        connection_weight=2,
        size_match_weight=1e-2,
        complexity_match_weight=1,
        entropy_weight=1,
        iterations=500,
        update_rate=0.01,
        gpu_gpu_distance=1,
        cpu_gpu_distance=10,
        move_models=True,
    ):
        """
        Assign models to different devices. In the scope of a single process.
        Assigner assumes all GPUs have the **same processing power**.

        Assignment is based on four aspects:

        1. Distance and model connections. Connection is usually indicated
           by the amount of data transmitted between two models.
        2. Compute complexity.
        3. Model size.
        4. Entropy.

        Four aspects are controlled by four weights:

        1. ``connection_weight``, assigner will try to reduce the total
           ``distance * connection`` if this weight is larger.
        2. ``size_match_weight``, this weight controls the total memory
           space used on a single device, only works if total assigned
           memory of models exceeds allowed device memory size
           (internally it uses a relu activation), the larger,
           the tighter and more restricted the fit.
        3. ``complexity_match_weight``, this weights balance the model
           computation cost across devices, assigner will try to even
           the ``computation cost / compute power`` ratio for each device
           if this weight is larger.
        4. ``entropy_weight``, this weight minimize the uncertainty of
           model placement probability, so ``model i`` will have a close to 1
           probability of locating on some ``device j`` if this weight is
           larger.

        Assignment uses gradient descent to compute the probability matrix
        of each ``model i`` locating on each available ``device j``.

        See Also:
            :class:`.ModelSizeEstimator`

        Note:
            When the sum of your model size is very close to the capacity of
            your device memory, `ModelAssigner` does not respond very well
            to the ``size_match_weight``, therefore, please consider about
            increasing ``model_size_multiplier`` or decreasing
            ``max_mem_ratio``.

        Args:
            models: Models to assign.
            model_connection: Connection weight between modules.
                **Must be positive**
            devices: Available devices.
            model_size_multiplier: Size multiplier of models, used to reserve
                enough space for models,
            max_mem_ratio: Maximum percent of memory allowed.
            cpu_weight: Weight of cpu. Relative to the computing power of one
                GPU. By default it is 0 so no computation will be performed on
                CPU. **Must be positive**
            connection_weight: Weight of connection between models.
            size_match_weight: Weight of size match.
            complexity_match_weight: Weight of complexity match.
            entropy_weight: Weight of entropy.
            iterations: Number of optimization iterations.
            update_rate: Learning rate of the adam optimizer.
            gpu_gpu_distance: Estimated distance cost between gpu-gpu.
                **Must be positive**
            cpu_gpu_distance: Estimated distance cost between cpu-gpu.
                **Must be positive**
            move_models: Whether to automatically move the models after
                assignment.
        """
        if devices is None:
            devices = [
                t.device(type="cuda", index=i)
                for i in GPUtil.getAvailable(order="load")
            ]
        else:
            devices = [t.device(d) for d in devices]
            available_devices = [
                t.device(type="cuda", index=i)
                for i in GPUtil.getAvailable(order="load")
            ]
            used_devices = []
            for dev in devices:
                if dev.type == "cuda" and dev not in available_devices:
                    default_logger.info(
                        f"Warning: device {dev} not available, removed."
                    )
                else:
                    used_devices.append(dev)
            devices = used_devices

        if not devices:
            devices = [t.device("cpu")]

        default_logger.info(f"Using these devices: {devices}")

        sizes = [
            ModelSizeEstimator(model, model_size_multiplier).estimate_size()
            for model in models
        ]
        device_size_capacity = []
        device_complexity_capacity = []

        gpus = GPUtil.getGPUs()
        for dev in devices:
            if dev.type == "cpu":
                device_size_capacity.append(
                    int(psutil.virtual_memory().available / 1024 ** 2) * max_mem_ratio
                )
                device_complexity_capacity.append(cpu_weight)
            elif dev.type == "cuda":
                device_size_capacity.append(gpus[dev.index].memoryFree * max_mem_ratio)
                device_complexity_capacity.append(1 - gpus[dev.index].load)

        if np.sum(np.array(sizes)) > np.sum(device_size_capacity):
            raise RuntimeError(
                f"Estimated model will use {np.sum(np.array(sizes)):.2f} MB, "
                f"but only have {np.sum(device_size_capacity):.2f} MB allowed memory "
                "in total."
            )

        # assign model to devices
        # using heuristic and gradient decent
        device_num = len(devices)
        model_num = len(models)

        # Important, the placement probability matrix! this matrix
        # describes the probability of placement of:
        # model i on device j
        placement = t.randn([model_num, device_num], requires_grad=True)

        optimizer = t.optim.Adam([placement], lr=update_rate)
        model_size = t.tensor(sizes, dtype=t.float).view([1, model_num])
        size_capacity = t.tensor(device_size_capacity, dtype=t.float).view(
            [1, device_num]
        )
        model_complexity = model_size

        # complexity_capacity is basically the estimated computing power
        # of devices.
        complexity_capacity = t.tensor(device_complexity_capacity, dtype=t.float).view(
            [1, device_num]
        )

        # model connection indicates the amount of data transmitted between
        # each pair of models, a weighted adjacency matrix.
        model_conn = t.zeros([model_num, model_num])

        for direction, conn in model_connection.items():
            model_conn[direction[0], direction[1]] = conn

        # device distance matrix
        device_distance = t.zeros([device_num, device_num])
        for i in range(device_num):
            for j in range(i):
                if (
                    devices[i].type == "cpu"
                    and devices[j].type == "cuda"
                    or devices[i].type == "cuda"
                    and devices[j].type == "cpu"
                ):
                    device_distance[i, j] = device_distance[j, i] = cpu_gpu_distance
                elif (
                    devices[i].type == "cuda"
                    and devices[j].type == "cuda"
                    and devices[i].index != devices[j].index
                ):
                    device_distance[i, j] = device_distance[j, i] = gpu_gpu_distance

        # optimize
        for _ in range(iterations):
            self.optimize_placement(
                optimizer,
                placement,
                model_size,
                size_capacity,
                model_complexity,
                complexity_capacity,
                model_conn,
                device_distance,
                connection_weight,
                size_match_weight,
                complexity_match_weight,
                entropy_weight,
            )
        self._assignment = [devices[d] for d in t.argmax(placement, dim=1).tolist()]
        if move_models:
            for model, ass_device in zip(models, self._assignment):
                model.to(ass_device)

    @property
    def assignment(self):
        """
        List[t.device]:
            Assigned devices for each model in your model list.
        """
        return self._assignment

    @staticmethod
    def optimize_placement(
        optimizer,
        placement: t.Tensor,
        model_size: t.Tensor,
        size_capacity: t.Tensor,
        model_complexity: t.Tensor,
        complexity_capacity: t.Tensor,
        model_connection: t.Tensor,
        device_distance: t.Tensor,
        connection_weight: float,
        size_match_weight: float,
        complexity_match_weight: float,
        entropy_weight: float,
    ):
        """
        Suppose there are n models to place and m devices available.

        Args:
            optimizer: optimizer of placement
            placement: shape ``[n, m]``
            model_size: shape ``[1, n]``
            size_capacity: shape ``[1, m]``
            model_complexity: shape ``[1, n]``
            complexity_capacity: shape ``[1, m]``
            model_connection: shape ``[n, n]``
            device_distance: shape ``[m, m]``
            connection_weight: Weight of connection between models.
            size_match_weight: Weight of size match.
            complexity_match_weight: Weight of complexity match.
            entropy_weight: weight of entropy.
        """
        placement = t.softmax(placement, dim=-1)
        model_num = placement.shape[0]

        norm_model_conn = model_connection / t.sum(model_connection)
        norm_dev_dist = device_distance / t.sum(device_distance)
        model_distance = t.einsum("ij,mn,jn->im", placement, placement, norm_dev_dist)
        # model distance to itself is 0
        model_distance[np.arange(model_num), np.arange(model_num)] = 0
        connection_cost = norm_model_conn * model_distance

        # sum(model size) < capacity
        size_match_cost = t.relu(
            t.einsum("ij,jk->ik", model_size, placement) - size_capacity
        )

        # match computing power percent
        norm_model_cmplx = model_complexity / t.sum(model_complexity)
        norm_cmplx_capacity = complexity_capacity / t.sum(complexity_capacity)
        cmplx_match_cost = (
            t.einsum("ij,jk->ik", norm_model_cmplx, placement) - norm_cmplx_capacity
        ) ** 2

        # entropy loss, prevent placement probability diffuse over devices
        entropy_cost = placement * placement.log()
        tmp = t.zeros_like(placement)
        entropy_cost = -t.where(placement > 0, entropy_cost, tmp).sum(dim=-1)

        total_cost = (
            t.mean(connection_cost) * connection_weight
            + t.mean(size_match_cost) * size_match_weight
            + t.mean(cmplx_match_cost) * complexity_match_weight
            + t.mean(entropy_cost) * entropy_weight
        )

        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()
