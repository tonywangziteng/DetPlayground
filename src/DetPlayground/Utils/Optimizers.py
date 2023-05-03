import os
from typing import Dict, List, Type, Union
from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim

class BaseOptimizer(ABC, optim.Optimizer):
    @abstractmethod
    def __init__(
        self, 
        model: nn.Module, 
        lr_start: float, 
        weight_decay: float, 
        **kargs
    ) -> None:
        super().__init__()

    def _get_param_groups(
        self,
        model: nn.Module, 
        weight_decay: float
    ) -> List[Dict]:
        """
        Return the paramter groups according to bias, bn, weight
        Please rewrite if needed.
        """
        # prepare parameter groups
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
                
        param_groups: List[Dict] = [
            {"params": pg0}, 
            {"params": pg1, "weight_decay": weight_decay}, 
            {"params": pg2}
        ]
        
        return param_groups

    
class AdamOptimizer(BaseOptimizer, optim.Adam):
    def __init__(
        self, 
        model: nn.Module,
        lr_start: float, 
        weight_decay: float, 
        **kargs: Dict
    ) -> None:
        param_groups = self._get_param_groups(model=model, weight_decay=weight_decay)
        optim.Adam.__init__(self, params=param_groups, lr=lr_start)
        

class SgdOptimizer(BaseOptimizer, optim.SGD):
    def __init__(
        self, 
        model: nn.Module,
        lr_start: float,
        momentum: float,
        weight_decay: float,
        **kargs: Dict
    ) -> None:
        param_groups = self._get_param_groups(model=model, weight_decay=weight_decay)
        optim.SGD.__init__(
            self,
            params=param_groups,
            lr=lr_start,
            momentum=momentum
        )
        

class RepSgdOptimizer(BaseOptimizer):
    pass
        
    
OPTIMIZER_COLLECTION: Dict[str, Type[BaseOptimizer]] = {
    "sgd": SgdOptimizer,
    "adam": AdamOptimizer,
    "rep_sgd_optimizer": RepSgdOptimizer
}

OptimizerTypes = Union[tuple(OPTIMIZER_COLLECTION.values())]

