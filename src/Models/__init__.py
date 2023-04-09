from typing import Dict
import torch.nn as nn
from Models.Backbones import Darknet

from .Yolo.YoloX import YoloX

model_collection: Dict[str, nn.Module] = {
    "YoloX-L": YoloX.YoloX_L
}