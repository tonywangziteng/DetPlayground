from typing import Dict, Type, Union
from Loss.BaseLossCalculator import BaseLossCalculator
from Loss.YoloXLossCalculator import YoloXLossCalculator

loss_calculator_type_union = Union[YoloXLossCalculator, BaseLossCalculator]

LOSS_CALCULATOR_COLLECTION: Dict[str, Type[loss_calculator_type_union]] = {
    "YoloX-Nano": YoloXLossCalculator
}
