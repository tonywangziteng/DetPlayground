from typing import List, Dict, Any

from abc import ABC
import torch
import torch.nn as nn

from DetPlayground.Models.Yolo.YoloX.YoloXHead import YoloXHead
from DetPlayground.Models.Yolo.YoloX.YoloXPan import YoloXPan
from DetPlayground.Models.CommonBlocks.BaseActivations import SiLU

class YoloX(nn.Module, ABC):
    def __init__(
        self, 
        depth_mul: float = 1.0, 
        width_mul: float = 1.0, 
        num_classes: int = 80, 
        strides: List[int] = [8, 16, 32], 
        in_channels: List[int] = [256, 512, 1024]
    ) -> None:
        super().__init__()
        self.backbone = YoloXPan(
            depth=depth_mul, width=width_mul
        )
        self.head = YoloXHead(
            num_classes=num_classes,
            width=width_mul, 
            strides=strides, 
            in_channels=in_channels
        )
        
    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        outputs: List[torch.Tensor] = self.head(fpn_outs)
        
        output_dict: Dict = {
            "raw_output": outputs, 
            "strides": self.head.strides
        }

        return output_dict


class YoloX_L(YoloX):
    def __init__(
        self, 
        args: Dict[str, Any]
    ) -> None:
        super().__init__(num_classes=args["num_classes"])


class YoloX_Nano(YoloX):
    def __init__(
        self, 
        args: Dict[str, Any]
    ) -> None:
        super().__init__(
            depth_mul=0.33, 
            width_mul=0.25, 
            num_classes=args["num_classes"]
        )
