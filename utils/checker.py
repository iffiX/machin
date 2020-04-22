import torch
import numpy as np


def check_shape(tensor, shape_list):
    shape = list(tensor.shape)
    if len(shape) != len(shape_list):
        raise RuntimeError("Shape length invalid, required {} dims, is {} dims".format(
            len(shape_list), len(shape)
        ))
    for s1, s2, dim in zip(shape, shape_list, range(len(shape_list))):
        if s2 > 0:
            if s1 != s2:
                raise RuntimeError("Shape invalid at dimension {}, required {}, is {}.".format(
                    dim, s2, s1
                ))


def check_nan(tensor, name):
    if torch.any(torch.isnan(tensor)):
        print(tensor)
        print(tensor.shape)
        print("Tensor {} contains nan!".format(name))
        raise RuntimeError("nan input")


def _gen_check_hook(writer, name, counter, dir, path="tmp/log/", param_freq=1, check_input=True, check_param=True):
    def check_hook(module, input, output):
        with torch.no_grad():
            args = [i for i in range(len(input))]
            for i in range(len(args)):
                if check_input and isinstance(input[i], (torch.Tensor, np.ndarray)):
                    check_nan(input[i], name + ".{}.{}_{}".format(dir, args[i], counter.get()))
                    #if dir == "backward":
                    #    torch.save(input[i], path + name + ".{}.{}_{}".format(dir, args[i], counter.get()))
                    writer.add_scalars(name + ".{}.{}".format(dir, args[i]),
                                       {"min": torch.min(input[i]),
                                        "max": torch.max(input[i]),
                                        "mean": torch.mean(input[i])}, counter.get())
                    writer.flush()
            if dir == "backward":
                if check_param and counter.get() % param_freq == 0:
                    for param_name, param_value in module.named_parameters():
                        check_nan(param_value, name + ".param.{}_{}".format(param_name, counter.get()))
                        writer.add_histogram(name + ".param.{}".format(param_name), param_value, counter.get())
                        writer.flush()

    return check_hook


def check_model(writer, model, global_counter, check_input=True, check_param=True, name=""):
    # register forward & backward hooks for all submodules
    for sub_name, sub_module in model.named_modules():
        if len([i for i in sub_module.modules()]) != 1:
            continue
        sub_module.register_forward_hook(
            _gen_check_hook(writer, "{}.{}".format(name, sub_name), global_counter, "forward",
                            check_input=check_input, check_param=check_param))
        sub_module.register_backward_hook(
            _gen_check_hook(writer, "{}.{}".format(name, sub_name), global_counter, "backward",
                            check_input=check_input, check_param=check_param))