from machin.parallel.assigner import ModelAssigner
from machin.model.nets.resnet import ResNet

if __name__ == "__main__":
    models = [
        ResNet(in_planes=9, depth=152, out_planes=1, out_pool_size=[20, 20])
        for _ in range(4)
    ]

    # create 4 ResNet as example models
    assigner = ModelAssigner(
        models=models,
        # `model_connection` (A, B): 1 means the amount of data trasmitted
        # from A to B is 1
        model_connection={(0, 1): 1, (2, 3): 1},
        # available devices
        devices=["cuda:0", "cpu"],
        model_size_multiplier=1,
        max_mem_ratio=0.7,
        # computing power compared to GPU, 0 means none, 1 means same
        cpu_weight=1,
        # larger connection weight will force assigner to place
        # models with larger quantities of data transmision on the
        # same device
        connection_weight=1e3,
        # try this and see what will happen
        # connection_weight=1e1,
        size_match_weight=1e-2,
        complexity_match_weight=10,
        entropy_weight=1,
        iterations=500,
        update_rate=0.01,
        gpu_gpu_distance=1,
        cpu_gpu_distance=10,
        move_models=False,
    )
    real_assignment = [str(dev) for dev in assigner.assignment]
    # should be "cuda:0", "cuda:0", "cpu", "cpu"
    # or "cpu", "cpu", "cuda:0", "cuda:0"
    print(f"Assignment: {real_assignment}")
