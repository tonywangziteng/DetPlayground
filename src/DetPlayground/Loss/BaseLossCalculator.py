from typing import List, Dict
from abc import ABC
from abc import abstractmethod

import torch


class BaseLossCalculator(ABC):
    def __init__(self, *args, **kargs) -> None:
        super().__init__()
        
    @abstractmethod
    def calculate_losses(
        self, 
        outputs: List[torch.Tensor], 
        targets: torch.Tensor,  # [bs, 120, 5]
    ) -> Dict:
        pass
