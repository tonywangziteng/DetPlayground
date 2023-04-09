from typing import Optional, Dict, Any

from abc import ABC
import torch.nn as nn

from Models.Yolo.YoloX.YoloXHead import YoloXHead
from Models.Yolo.YoloX.YoloXPan import YoloXPan
from Models.CommonBlocks.BaseActivations import SiLU

class YoloX(nn.Module, ABC):
    def __init__(
        self, 
        backbone: Optional[nn.Module] = None, 
        head: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = YoloXPan()
        if head is None:
            head = YoloXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)

        return outputs

class YoloX_L(YoloX):
    def __init__(
        self, 
        args: Dict[str, Any]
    ) -> None:
        head = YoloXHead(num_classes=args["num_classes"])
        super().__init__(head=head)
