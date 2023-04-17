from typing import Dict, Union
import torch.nn as nn
from Models.Backbones import Darknet

from .Yolo.YoloX import YoloX


model_collection: Dict[str, YoloX.YoloX] = {
    "YoloX-L": YoloX.YoloX_L, 
    "YoloX-Nano": YoloX.YoloX_Nano
}