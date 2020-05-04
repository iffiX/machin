import torch.nn as nn


class NeuralNetworkModule(nn.Module):
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
                return [p.device for p in self.input_module.parameters()][0]
            else:
                return self.input_module.input_device

    @property
    def output_device(self):
        if self.output_module is None and self.input_module is None:
            raise RuntimeError("Output module not set.")
        elif self.output_module is not None:
            if not isinstance(self.output_module, NeuralNetworkModule):
                return [p.device for p in self.output_module.parameters()][0]
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


class NeuralNetworkWrapper:
    """
    This wrapper is used to wrap a vanilla nn.Module, and provide
    input_device and output_device to frameworks
    """

    def __init__(self, wrapped_module: nn.Module, input_device, output_device):
        self.wrapped_module = wrapped_module
        self.input_device = input_device
        self.output_device = output_device

    def __getattr__(self, item):
        # if access some attribute that could not be found
        return self.wrapped_module.__getattribute__(item)

    def __setattr__(self, key, value):
        if key not in {"wrapped_module", "input_device", "output_device"}:
            return self.wrapped_module.__setattr__(key, value)
        else:
            super(NeuralNetworkWrapper, self).__setattr__(key, value)

    def __call__(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)
