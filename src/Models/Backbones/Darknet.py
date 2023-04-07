from typing import Any, Optional, Callable, List
from abc import ABC
from abc import abstractmethod
import torch
import torch.nn as nn

from Models.CommonBlocks.BaseConvs import BaseConv
from Models.CommonBlocks.BaseConvs import DWConv
from Models.CommonBlocks.BaseBlocks import ResidualBlockYoloX as ResLayer
from Models.CommonBlocks.BaseBlocks import CSPLayer
from Models.CommonBlocks.BaseBlocks import Focus
from Models.CommonBlocks.BaseBlocks import SPPBottleneck

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, kernel_size=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
    

class Darknet(nn.Module, ABC):
    def __init__(
        self,
        num_blocks: List[int],
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, kernel_size=3, stride=1, act=nn.LeakyReLU),
            *self.make_stage(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = self.make_stage(in_channels, num_blocks[0], stride=2)
        
        in_channels *= 2  # 128
        self.dark3 = self.make_stage(in_channels, num_blocks[1], stride=2)
        
        in_channels *= 2  # 256
        self.dark4 = self.make_stage(in_channels, num_blocks[2], stride=2)
        
        in_channels *= 2  # 512
        self.dark5 = nn.Sequential(
            self.make_stage(in_channels, num_blocks[3], stride=2),
            self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )
        
    def make_stage(self, in_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            BaseConv(in_channels, in_channels * 2, kernel_size=3, stride=stride, act=nn.LeakyReLU),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        )

    def make_spp_block(self, filters_list, in_filters) -> nn.Sequential:
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act=nn.LeakyReLU),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=nn.LeakyReLU),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation=nn.LeakyReLU,
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=nn.LeakyReLU),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act=nn.LeakyReLU),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}



        


class Darknet53(Darknet):
    def __init__(
        self, 
        in_channels=3, 
        stem_out_channels=32, 
        out_features=("dark3", "dark4", "dark5")
    ):
        super().__init__(
            num_blocks = [2, 8, 8, 4], 
            in_channels = in_channels, 
            stem_out_channels = stem_out_channels, 
            out_features = out_features
        )
        
        
class Darknet21(Darknet):
    def __init__(
        self, 
        in_channels=3, 
        stem_out_channels=32, 
        out_features=("dark3", "dark4", "dark5")
    ):
        super().__init__(
            num_blocks = [1, 2, 2, 1], 
            in_channels = in_channels, 
            stem_out_channels = stem_out_channels, 
            out_features = out_features
        )
