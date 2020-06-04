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


def safe_call(model, *named_args, required_argument=()):
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
        required_argument: A list/tuple of required arguments' name.

    Returns:
        Whatever returned by your module.
    """
    if (not hasattr(model, "input_device") or
            not hasattr(model, "output_device")):
        raise RuntimeError("Wrap your model of type nn.Module with one of: \n"
                           "1. StaticModuleWrapper from models.models.base \n"
                           "2. DynamicModuleWrapper from models.models.base \n"
                           "Or construct your own module & model with: \n"
                           "NeuralNetworkModule from models.models.base")
    input_device = model.input_device
    args = inspect.getfullargspec(model.forward).args
    args_dict = {}
    if any(arg not in args for arg in required_argument):
        missing = []
        for arg in required_argument:
            if arg not in args:
                missing.append(arg)
        raise RuntimeError("Model missing required argument field(s): {}, "
                           "check your store_observe() function.".format(missing))
    for na in named_args:
        for k, v in na.items():
            if k in args:
                if torch.is_tensor(v):
                    args_dict[k] = v.to(input_device)
                else:
                    args_dict[k] = v
    return model(**args_dict)


def assert_output_is_probs(tensor):
    if tensor.dim() == 2 and \
            torch.all(torch.abs(torch.sum(tensor, dim=1) - 1.0) < 1e-5) and \
            torch.all(tensor > 0):
        return
    else:
        raise RuntimeError("Input tensor is not a probability tensor, it must "
                           "have a dimension of 2, a sum of 1.0 for each row in"
                           " dimension 1, and a positive value for each "
                           "element.")
