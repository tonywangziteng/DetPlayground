from typing import Any, Optional, Callable

import torch 
import torch.nn as nn

from Models.CommonBlocks.BaseConvs import BaseConv
from Models.CommonBlocks.BaseConvs import DWConv
from Models.CommonBlocks.BaseConvs import Identity


class ResidualBlock(nn.Module):
    """
    residual block in the original paper
    """
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        act: Callable[[bool], nn.Module] = nn.ReLU,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self._conv1 = BaseConv(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            kernel_size=3, 
            stride=stride, 
            act=act
        )
        self._conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            padding=1
        )
        self._bn = nn.BatchNorm2d(out_channels)
        self._act = act(inplace=True)
        self._downsample = downsample
        self._identity = Identity()

    def forward(self, x: torch.Tensor):
        identity = self._identity(x)
        out = self._conv1(x)
        out = self._bn(self._conv2(out))
        if self._downsample is not None:
            identity = self._downsample(identity)
        out = out + identity
        return self._act(out)


class ResidualBlockPreAct(nn.Module):
    """
    pre-activations with batch normalizations, usually achieves better result
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        act: Callable[[bool], nn.Module] = nn.ReLU,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.act = act(inpulace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=stride, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            padding=1
        )
        self.downsample = downsample
        self.identity = Identity()

    def forward(self, x: torch.Tensor):
        identity = self.identity(x)

        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        return out

class BottleneckResnet(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        hidden_channels: int, 
        stride: int = 1, 
        act: Callable[[bool], nn.Module] = nn.ReLU,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = BaseConv(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            kernel_size=1, 
            stride=1, 
            act=act
        )
        
        self.conv2 = BaseConv(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            kernel_size=3, 
            stride=stride, 
            act=act
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        self.identity = Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)

        out = self.conv2(self.conv1(x))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
        
class Bottleneck(nn.Module):
    """
    Standard bottleneck (from YoloX)
    ConvBnRelu + Conv + (skip connection)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int, 
        out_channels: int,
        skip_connection: bool = True,
        depthwise: bool = False, 
        act: Callable[[bool], nn.Module] = nn.SiLU,
    ):
        """
        Args:
            in_channels (int): input channel number
            hidden_channels (int): hidden channel number
            out_channels (int): output channel number
            shortcut (bool, optional): Skip connection, only can be used when in_channels==out_channels. Defaults to True.
            depthwise (bool, optional): Use depthwise convolution or not. Defaults to False.
            act (Callable[[bool], nn.Module], optional): Activation function to use. Defaults to nn.SiLU.
        """
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_skip_connection = skip_connection and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_skip_connection:
            y = y + x
        return y
    
    