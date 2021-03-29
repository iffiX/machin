from .base import NeuralNetworkModule, dynamic_module_wrapper, static_module_wrapper

from .resnet import ResNet

from . import base
from . import resnet

__all__ = [
    "NeuralNetworkModule",
    "dynamic_module_wrapper",
    "static_module_wrapper",
    "ResNet",
    "base",
    "resnet",
]
