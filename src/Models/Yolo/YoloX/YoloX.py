from typing import Optional, Dict, Any

from abc import ABC
import torch.nn as nn

from Models.Yolo.YoloX.YoloXHead import YoloXHead
from Models.Yolo.YoloX.YoloXPan import YoloXPan
from Models.CommonBlocks.BaseActivations import SiLU

class YoloX(nn.Module, ABC):
    def __init__(
        self, 
        depth_mul: float = 1.0, 
        width_mul: float = 1.0, 
        num_classes: int = 80
    ) -> None:
        super().__init__()
        self.backbone = YoloXPan(
            depth=depth_mul, width=width_mul
        )
        self.head = YoloXHead(
            num_classes=num_classes, width=width_mul
        )

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)

        return outputs
    
    @property
    def strides(self):
        return self.head.strides

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
