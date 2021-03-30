import torch as t
import numpy as np
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from .base import NeuralNetworkModule


def conv1x1(in_planes, out_planes, stride=1):
    """
    Create a 1x1 2d convolution block
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """
    Create a 3x3 2d convolution block
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv5x5(in_planes, out_planes, stride=2):
    """
    Create a 5x5 2d convolution block
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False
    )


def none_norm(*_, **__):
    return nn.Sequential()


def cfg(depth, norm="none"):
    depth_lst = [18, 34, 50, 101, 152]
    if depth not in depth_lst:
        raise ValueError(
            "Error : Resnet depth should be either " "18, 34, 50, 101, 152"
        )
    if norm == "batch":
        cfg_dict = {
            "18": (BasicBlock, [2, 2, 2, 2], {}),
            "34": (BasicBlock, [3, 4, 6, 3], {}),
            "50": (Bottleneck, [3, 4, 6, 3], {}),
            "101": (Bottleneck, [3, 4, 23, 3], {}),
            "152": (Bottleneck, [3, 8, 36, 3], {}),
        }
    elif norm == "weight":
        cfg_dict = {
            "18": (BasicBlockWN, [2, 2, 2, 2], {}),
            "34": (BasicBlockWN, [3, 4, 6, 3], {}),
            "50": (BottleneckWN, [3, 4, 6, 3], {}),
            "101": (BottleneckWN, [3, 4, 23, 3], {}),
            "152": (BottleneckWN, [3, 8, 36, 3], {}),
        }
    elif norm == "none":
        cfg_dict = {
            "18": (BasicBlock, [2, 2, 2, 2], {"norm_layer": none_norm}),
            "34": (BasicBlock, [3, 4, 6, 3], {"norm_layer": none_norm}),
            "50": (Bottleneck, [3, 4, 6, 3], {"norm_layer": none_norm}),
            "101": (Bottleneck, [3, 4, 23, 3], {"norm_layer": none_norm}),
            "152": (Bottleneck, [3, 8, 36, 3], {"norm_layer": none_norm}),
        }
    else:
        raise ValueError(f'Invalid normalization method: "{norm}"')
    return cfg_dict[str(depth)]


class BasicBlock(NeuralNetworkModule):
    # Expansion parameter, output will have "expansion * in_planes" depth.
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, norm_layer=None, **__):
        """
        Create a basic block of resnet.

        Args:
            in_planes:  Number of input planes.
            out_planes: Number of output planes.
            stride:     Stride of convolution.
        """
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = conv3x3(out_planes, self.expansion * out_planes)
        self.bn2 = norm_layer(self.expansion * out_planes)
        # Create a shortcut from input to output.
        # An empty sequential structure means no transformation
        # is made on input X.
        self.shortcut = nn.Sequential()

        self.set_input_module(self.conv1)

        # A convolution is needed if we cannot directly add input X to output
        # BatchNorm2d produces NaN gradient?
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * out_planes, stride),
                norm_layer(self.expansion * out_planes),
            )

    def forward(self, x):
        out = t.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = t.relu(out)
        return out


class Bottleneck(NeuralNetworkModule):
    # Expansion parameter, output will have "expansion * in_planes" depth.
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, norm_layer=None, **__):
        """
        Create a bottleneck block of resnet.

        Args:
            in_planes:  Number of input planes.
            out_planes: Number of output planes.
            stride:     Stride of convolution.
        """
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__()
        self.conv1 = conv1x1(in_planes, out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride=stride)
        self.conv3 = conv1x1(out_planes, self.expansion * out_planes)
        self.bn1 = norm_layer(out_planes)
        self.bn2 = norm_layer(out_planes)
        self.bn3 = norm_layer(self.expansion * out_planes)

        self.shortcut = nn.Sequential()

        self.set_input_module(self.conv1)

        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * out_planes, stride),
                norm_layer(self.expansion * out_planes),
            )

    def forward(self, x):
        out = t.relu(self.bn1(self.conv1(x)))
        out = t.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = t.relu(out)
        return out


class BasicBlockWN(NeuralNetworkModule):
    """
    Basic block with weight normalization
    """

    # Expansion parameter, output will have "expansion * in_planes" depth.
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, **__):
        """
        Create a basic block of resnet.

        Args:
            in_planes:  Number of input planes.
            out_planes: Number of output planes.
            stride:     Stride of convolution.
        """
        super().__init__()
        self.conv1 = weight_norm(conv3x3(in_planes, out_planes, stride))
        self.conv2 = weight_norm(conv3x3(out_planes, self.expansion * out_planes))
        # Create a shortcut from input to output.
        # An empty sequential structure means no transformation
        # is made on input X.
        self.shortcut = nn.Sequential()

        self.set_input_module(self.conv1)

        # A convolution is needed if we cannot directly add input X to output.
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                weight_norm(conv1x1(in_planes, self.expansion * out_planes, stride)),
            )

    def forward(self, x):
        out = t.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = t.relu(out)
        return out


class BottleneckWN(NeuralNetworkModule):
    """
    Bottleneck block with weight normalization
    """

    # expansion parameter, output will have "expansion * in_planes" depth
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, **__):
        """
        Create a bottleneck block of resnet.

        Args:
            in_planes:  Number of input planes.
            out_planes: Number of output planes.
            stride:     Stride of convolution.
        """
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        )
        self.conv2 = weight_norm(
            nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        )
        self.conv3 = weight_norm(
            nn.Conv2d(
                out_planes, self.expansion * out_planes, kernel_size=1, bias=False
            )
        )

        self.shortcut = nn.Sequential()

        self.set_input_module(self.conv1)

        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                weight_norm(conv1x1(in_planes, self.expansion * out_planes, stride)),
            )

    def forward(self, x):
        out = t.relu(self.conv1(x))
        out = t.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = t.relu(out)

        return out


class ResNet(NeuralNetworkModule):
    def __init__(
        self,
        in_planes: int,
        depth: int,
        out_planes: int,
        out_pool_size=(1, 1),
        norm="none",
    ):
        """
        Create a resnet of specified depth.

        Args:
            in_planes:  Number of input planes.
            depth: Depth of resnet. Could be one of ``18, 34, 50, 101, 152``.
            out_planes: Number of output planes.
            out_pool_size: Size of pooling output
            norm: Normalization method, could be one of "none", "batch" or
                "weight".
        """
        super().__init__()
        self.norm = norm
        self.depth = depth
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.out_pool_size = out_pool_size

        self._cur_in_planes = 64

        block, num_blocks, kw = cfg(depth, norm)

        self.conv1 = nn.Conv2d(
            in_planes, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if norm == "batch":
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, kw)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, kw)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, kw)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, kw)
            self.base = nn.Sequential(
                self.conv1,
                self.bn1,
                nn.ReLU(),
                self.maxpool,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
            )
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, kw)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, kw)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, kw)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, kw)
            self.base = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                self.maxpool,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
            )
        self.fc = nn.Linear(512 * out_pool_size[0] * out_pool_size[1], out_planes)

        self.set_input_module(self.conv1)

    def _make_layer(self, block, planes, num_blocks, stride, kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self._cur_in_planes, planes, stride, **kwargs))
            self._cur_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        assert x.shape[2] == 224 and x.shape[3] == 224
        x = self.base(x)
        kernel_size = (
            np.int(np.floor(x.size(2) / self.out_pool_size[0])),
            np.int(np.floor(x.size(3) / self.out_pool_size[1])),
        )
        x = nn.functional.avg_pool2d(x, kernel_size)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
