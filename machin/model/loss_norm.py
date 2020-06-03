import numpy as np
import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class LossNorm:
    _func = {}
    _args = {}
    _kwargs = {}
    _weight = {}
    _scale = {}
    _bias = {}
    _history = {}

    def __init__(self, min_deriv=1e-3, max_history=1000):
        if min_deriv > 0.25:
            raise RuntimeError("Min derivation cannot be larger than 0.25")
        elif min_deriv <= 0:
            raise RuntimeError("Min derivation must be larger than 0")
        self.range = np.log(2 * min_deriv / ((1 - 2 * min_deriv) - np.sqrt(1 - 4 * min_deriv) + 1e-6))
        self.max_history = max_history

    def add_loss(self, loss_name, loss_func, *args, **kwargs):
        self._func[loss_name] = loss_func
        self._args[loss_name] = args
        self._kwargs[loss_name] = kwargs
        self._weight[loss_name] = 1
        self._scale[loss_name] = 1
        self._bias[loss_name] = 1
        self._history[loss_name] = []

    def remove_loss(self, loss_name):
        if loss_name in self._func:
            self._func.pop(loss_name)
            self._args.pop(loss_name)
            self._kwargs.pop(loss_name)
            self._weight.pop(loss_name)
            self._scale.pop(loss_name)
            self._bias.pop(loss_name)
            self._history.pop(loss_name)

    def get_total_loss(self):
        total_loss = 0
        for n, loss_func in self._func.items():
            new_loss = self._func[n](*self._args[n], **self._kwargs[n])
            self.__update_scale(n, new_loss)
            capped_loss = sigmoid((new_loss - self._bias[n]) / (self._scale[n] + 1e-6))
            total_loss += self._weight[n] * capped_loss
        return total_loss

    def set_loss_weight(self, loss_name, weight):
        weight = float(weight)
        if loss_name in self._func:
            self._weight[loss_name] = weight

    def __update_scale(self, loss_name, loss):
        loss = float(loss)
        history = self._history[loss_name]
        history.append(loss)
        if len(history) > self.max_history:
            history.pop(0)
        low, high = np.percentile(history, [10, 90])
        self._bias[loss_name] = (high + low) / 2
        self._scale[loss_name] = (high - low) / self.range
        print("name={}, bias={}, scale={}".format(loss_name, self._bias[loss_name], self._scale[loss_name]))
