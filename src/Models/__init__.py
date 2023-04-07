from typing import Dict
import torch.nn as nn
from Models.Backbones import Darknet

model_collection: Dict[str, nn.Module] = {
    "YoloX-L": Darknet
}