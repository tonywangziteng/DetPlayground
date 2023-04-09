from typing import Any, Optional, Callable, List

import torch 
import torch.nn as nn

from Models.CommonBlocks.BaseConvs import BaseConv
from Models.CommonBlocks.BaseConvs import DWConv
from Models.CommonBlocks.BaseConvs import Identity
from Models.CommonBlocks.BaseActivations import SiLU
   

class ResidualBlock(nn.Module):
    """
    residual block in the original paper
    """
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        act: nn.Module = nn.ReLU,
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
        act: nn.Module = nn.ReLU,
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

class ResidualBlockYoloX(nn.Module):
    """
    two conv blocks with a skip connection, 
    Hidden layer channel is in_channel//2 (some kind of bottleneck)
    activate before merging skip connection.  
    """
    def __init__(
        self,
        in_channels: int, 
        act: nn.Module = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, kernel_size=1, stride=1, act=act
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, kernel_size=3, stride=1, act=act
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out

class BottleneckResnet(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        hidden_channels: int, 
        stride: int = 1, 
        act: nn.Module = nn.ReLU,
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
        act: nn.Module = nn.SiLU,
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
    
class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x
    
class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 1, 
        stride: int = 1, 
        act: nn.Module = SiLU
    ):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
    
class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        bottleneck_num: int=1,
        skip_connection: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: nn.Module = SiLU,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                in_channels=hidden_channels, 
                hidden_channels=hidden_channels, 
                out_channels=hidden_channels, 
                skip_connection=skip_connection, 
                depthwise=depthwise, 
                act=act
            )
            for _ in range(bottleneck_num)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)
    
    

