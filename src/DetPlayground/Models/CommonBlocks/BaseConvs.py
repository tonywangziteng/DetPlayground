from typing import Any, Optional, Callable

import torch 
import torch.nn as nn

class Identity:
    def __call__(self, x):
        return x

class BaseConv(nn.Module):
    """
    Conv2d + norm + activation
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        groups: int = 1,
        bias: bool = False,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        padding: int = (kernel_size - 1) // 2
        self._conv2d = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            bias=bias
        )

        self._norm = nn.BatchNorm2d(out_channels)
        self._activation = act(inplace=True)

    def forward(self, x: torch.Tensor):
        return self._activation(self._norm(self._conv2d(x)))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        ksize: int, 
        stride: int = 1, 
        act: nn.Module = nn.SiLU
    ):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
