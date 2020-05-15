import torch
import torch.nn as nn


class NeuralNetworkModule(nn.Module):
    """
    Note: input device and output device are determined by module parameters,
          your input module / output module should not store parameters on
          more than one devices, and you also should not move your output to
          other devices other than your parameter storage device in forward().
    """
    def __init__(self):
        super(NeuralNetworkModule, self).__init__()
        self.input_module = None
        self.output_module = None

    def set_input_module(self, input_module):
        self.input_module = input_module

        if not isinstance(input_module, NeuralNetworkModule):
            if isinstance(input_module, nn.Sequential):
                input_module = self.find_child(input_module, True)
            if len({p.device for p in input_module.parameters()}) > 1:
                raise RuntimeError("Input module must be another NeuralNetworkModule "
                                   "or locate on one single device.")

    def set_output_module(self, output_module):
        self.output_module = output_module

        if not isinstance(output_module, NeuralNetworkModule):
            if isinstance(output_module, nn.Sequential):
                output_module = self.find_child(output_module, False)
            if len({p.device for p in output_module.parameters()}) > 1:
                raise RuntimeError("Output module must be another NeuralNetworkModule "
                                   "or locate on one single device.")

    @property
    def input_device(self):
        if self.input_module is None:
            raise RuntimeError("Input module not set.")
        else:
            if not isinstance(self.input_module, NeuralNetworkModule):
                set = {p.device for p in self.input_module.parameters()}
                if len(set) != 1:
                    raise RuntimeError("This input module contains parameters on different devices, "
                                       "please consider about splitting it.")
                else:
                    return list(set)[0]
            else:
                return self.input_module.input_device

    @property
    def output_device(self):
        if self.output_module is None and self.input_module is None:
            raise RuntimeError("Output module not set.")
        elif self.output_module is not None:
            if not isinstance(self.output_module, NeuralNetworkModule):
                set = {p.device for p in self.output_module.parameters()}
                if len(set) != 1:
                    raise RuntimeError("This output module contains parameters on different devices, "
                                       "please consider about splitting it.")
                else:
                    return list(set)[0]
            else:
                return self.output_module.output_device
        else:
            # input device is the same as output device
            return self.input_device

    @staticmethod
    def find_child(seq, is_first=True):
        if isinstance(seq, nn.Sequential):
            if is_first:
                return NeuralNetworkModule.find_child(seq[0], is_first)
            else:
                return NeuralNetworkModule.find_child(seq[-1], is_first)
        else:
            return seq

    def forward(self, *_):
        pass


def StaticModuleWrapper(wrapped_module: nn.Module, input_device, output_device):
    """
    Wrapped module could locate on multiple devices, but must not be moved.
    """
    wrapped_module.input_device = input_device
    wrapped_module.output_device = output_device
    return wrapped_module


def DynamicModuleWrapper(wrapped_module: nn.Module):
    """
    Wrapped module must locate on one single device, but could be moved around.
    """
    wrapper = NeuralNetworkModule()
    wrapper.add_module("wrapped_module", wrapped_module)
    wrapper.set_input_module(wrapped_module)
    wrapper.set_output_module(wrapped_module)
    wrapper.forward = wrapped_module.forward
    return wrapper


