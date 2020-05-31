from typing import List
from .helper_classes import Counter
import torch as t
import numpy as np


def check_shape(tensor: t.Tensor, required_shape: List[int]):
    shape = list(tensor.shape)
    if shape != required_shape:
        raise RuntimeError(
            "Tensor has invalid shape, required shape {}, is {}"
                .format(required_shape, shape)
        )


def check_nan(tensor, name):
    if t.any(t.isnan(tensor)):
        print(tensor)
        print(tensor.shape)
        print("Tensor {} contains nan!".format(name))
        raise RuntimeError("nan input")


def _gen_check_hook(writer, name, counter, dir, path="tmp/log/",
                    param_freq=100, check_input=True, check_param=True):
    def check_hook(module, hook_input, hook_output):
        with t.no_grad():
            arg_idx = [i for i in range(len(hook_input))]
            for idx in arg_idx:
                if check_input and isinstance(hook_input[idx],
                                              (t.Tensor, np.ndarray)):
                    check_nan(hook_input[idx],
                              name + ".{}.arg.{}_{}"
                              .format(dir, idx, counter.get()))
                    # if dir == "backward":
                    #    t.save(input[i], path + name + ".{}.{}_{}".format(dir, idx[i], counter.get()))
                    writer.add_scalars(name + ".{}.arg.{}"
                                       .format(dir, idx),
                                       {"min": t.min(hook_input[idx]),
                                        "max": t.max(hook_input[idx]),
                                        "mean": t.mean(hook_input[idx])},
                                       counter.get())
                    writer.flush()
            if dir == "backward":
                if check_param and counter.get() % param_freq == 0:
                    for param_name, param_value in module.named_parameters():
                        check_nan(param_value,
                                  name + ".param.{}_{}".format(param_name,
                                                               counter.get()))
                        writer.add_histogram(
                            name + ".param.{}".format(param_name), param_value,
                            counter.get())
                        writer.flush()

    return check_hook


def _model_backward_hook(model, _0, _1):
    getattr(model, "_utils_checker_back_counter").count()


def check_model(writer, model,
                check_input=True, check_param=True,
                name=""):
    setattr(model, "_utils_checker_back_counter", Counter())
    counter = getattr(model, "_utils_checker_back_counter")
    # register forward & backward hooks for all submodules
    for sub_name, sub_module in model.named_modules():
        if len([i for i in sub_module.modules()]) != 1:
            continue

        sub_module.register_forward_hook(
            _gen_check_hook(writer, "{}.{}".format(name, sub_name), counter,
                            "forward",
                            check_input=check_input, check_param=check_param))

        sub_module.register_backward_hook(
            _gen_check_hook(writer, "{}.{}".format(name, sub_name), counter,
                            "backward",
                            check_input=check_input, check_param=check_param))
