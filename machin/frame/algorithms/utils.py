from machin.utils.logging import default_logger
from machin.model.nets.base import static_module_wrapper
import inspect
import torch
import torch.nn as nn


def soft_update(target_net: nn.Module,
                source_net: nn.Module,
                update_rate) -> None:
    """
    Soft update target network's parameters.

    Args:
        target_net: Target network to be updated.
        source_net: Source network providing new parameters.
        update_rate: Update rate.

    Returns:
        None
    """
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(),
                                       source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - update_rate) +
                param.data.to(target_param.device) * update_rate
            )


def hard_update(target_net: nn.Module,
                source_net: nn.Module) -> None:
    """
    Hard update (directly copy) target network's parameters.

    Args:
        target_net: Target network to be updated.
        source_net: Source network providing new parameters.
    """

    for target_buffer, buffer in zip(target_net.buffers(),
                                     source_net.buffers()):
        target_buffer.data.copy_(buffer.data)
    for target_param, param in zip(target_net.parameters(),
                                   source_net.parameters()):
        target_param.data.copy_(param.data)


def determine_device(model):
    devices = set()
    for k, v in model.named_parameters():
        devices.add(str(v.device))
    return list(devices)


def safe_call(model, *named_args):
    """
    Call a model and discard unnecessary arguments. safe_call will automatically
    move tensors in named_args to the input device of the model

    Any input tensor in named_args must not be contained inside any container,
    such as list, dict, tuple, etc. Because they will be automatically moved
    to the input device of the specified model.

    Args:
        model: Model to be called, must be a wrapped nn.Module or an instance of
               NeuralNetworkModule.
        named_args: A dictionary of argument, key is argument's name, value is
                    argument's value.

    Returns:
        Whatever returned by your module. If result is not a tuple, always
        wrap results inside a tuple
    """
    org_model = None
    if isinstance(model,
                  (nn.parallel.DistributedDataParallel,
                   nn.parallel.DataParallel)):
        org_model = model
        model = model.module
    if (not hasattr(model, "input_device") or
            not hasattr(model, "output_device")):
        # try to automatically determine the input & output device of the model
        model_type = type(model)
        device = determine_device(model)
        if len(device) > 1:
            raise RuntimeError("Failed to automatically determine i/o device "
                               "of your model: {}\n"
                               "Detected multiple devices: {}\n"
                               "You need to manually specify i/o device of "
                               "your model.\n"
                               "Wrap your model of type nn.Module with one "
                               "of: \n"
                               "1. static_module_wrapper "
                               "from machin.model.nets.base \n"
                               "1. dynamic_module_wrapper "
                               "from machin.model.nets.base \n"
                               "Or construct your own module & model with: \n"
                               "NeuralNetworkModule from machin.model.nets.base"
                               .format(model_type, device))
        else:
            # assume that i/o devices are the same as parameter device
            # print a warning
            default_logger.warning("You have not specified the i/o device of "
                                   "your model {}, automatically determined and"
                                   " set to: {}\n"
                                   "The framework is not responsible for any "
                                   "un-matching device issues caused by this "
                                   "operation.".format(model_type, device[0]))
            model = static_module_wrapper(model, device[0], device[0])

    input_device = model.input_device
    arg_spec = inspect.getfullargspec(model.forward)
    # exclude self in arg_spec.args
    args = arg_spec.args[1:] + arg_spec.kwonlyargs
    if arg_spec.defaults is not None:
        args_with_defaults = args[-len(arg_spec.defaults):]
    else:
        args_with_defaults = []
    required_args = (set(args) - set(args_with_defaults) -
                     set(arg_spec.kwonlydefaults.keys()
                         if arg_spec.kwonlydefaults is not None
                         else []))
    args_dict = {}

    # fill in args
    for na in named_args:
        for k, v in na.items():
            if k in args:
                if torch.is_tensor(v):
                    args_dict[k] = v.to(input_device)
                else:
                    args_dict[k] = v

    # check for necessary args
    missing = required_args - set(args_dict.keys())
    if len(missing) > 0:
        raise RuntimeError("\n"
                           "The signature of the forward function of Model {} "
                           "is {}\n"
                           "Missing required arguments: {}, "
                           "check your storage functions."
                           .format(type(model), required_args, missing))

    if org_model is not None:
        result = org_model(**args_dict)
    else:
        result = model(**args_dict)

    if isinstance(result, tuple):
        return result
    else:
        return (result,)


def safe_return(result):
    if len(result) == 1:
        return result[0]
    else:
        return result


asserted_output_is_probs = False


def assert_output_is_probs(tensor):
    global asserted_output_is_probs
    if asserted_output_is_probs:
        return
    asserted_output_is_probs = True

    if tensor.dim() == 2 and \
            torch.all(torch.abs(torch.sum(tensor, dim=1) - 1.0) < 1e-5) and \
            torch.all(tensor > 0):
        return
    else:
        print(tensor)
        raise ValueError("Input tensor is not a probability tensor, it must "
                         "have 2 dimensions (0 and 1), a sum of 1.0 for each "
                         "row in dimension 1, and a positive value for each "
                         "element.")
