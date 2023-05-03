from typing import Dict, Type, Union
import torch.nn as nn
from Models.Backbones import Darknet

from .Yolo.YoloX import YoloX


model_type_union = Union[
    YoloX.YoloX_L, 
    YoloX.YoloX_Nano
]
MODEL_COLLECTION: Dict[str, Type[model_type_union]] = {
    "YoloX-L": YoloX.YoloX_L, 
    "YoloX-Nano": YoloX.YoloX_Nano
}