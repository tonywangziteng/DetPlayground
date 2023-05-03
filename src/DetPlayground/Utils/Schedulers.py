from typing import Dict, Optional, List, Type, Any
import warnings
import math

from abc import ABC
from abc import abstractmethod

from DetPlayground.Utils.Optimizers import BaseOptimizer

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class BaseScheduler(ABC, _LRScheduler):
    """
    The base class for every scheduler
    """
    def __init__(
        self, 
        optimizer: BaseOptimizer, 
        epoch_num: int, 
        last_epoch: int = -1,
        verbose: bool = False,
        **kargs
    ) -> None:
        _LRScheduler.__init__(self, optimizer, last_epoch=last_epoch, verbose=verbose)
        self._epoch_num = epoch_num
        
    @property
    def epoch_num(self) -> int:
        return self._epoch_num
    
    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return self.optimizer.param_groups
        

class CosineSchedulerWarmup(BaseScheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        iter_per_epoch: int, 
        epoch_num: int,
        last_epoch: int = -1, 
        verbose: bool = False, 
        lr_warmup_ratio: float = 0.1,
        lr_warmup_epoch: int = 0,
        lr_esaturate_ratio: float = 0.1,
        lr_saturate_epoch: int = 0,
        **kargs: Dict
    ) -> None:
        self._iter_per_epoch = iter_per_epoch
        
        self._lr_warmup_iter = lr_warmup_epoch * iter_per_epoch
        self._lr_saturate_iter = lr_saturate_epoch * iter_per_epoch
        self._all_iter: int = epoch_num * iter_per_epoch
        
        self._lr_start = optimizer.param_groups[0]["lr"]
        self._lr_warmup_start = lr_warmup_ratio * self._lr_start
        self._lr_saturate = lr_esaturate_ratio * self._lr_start
        super().__init__(optimizer, last_epoch, verbose, **kargs)
        
    def get_lr(self):
        step_count = self._step_count - 1
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        if self._step_count < self._lr_warmup_iter:
            # warmup phase
            cur_lr = (self._lr_start - self._lr_warmup_start) * pow(
                step_count / float(self._lr_warmup_iter), 2
            ) + self._lr_warmup_start
        elif self._step_count >= self._all_iter - self._lr_saturate_iter:
            # saturated phase
            cur_lr = self._lr_end
        else:
            cur_lr = self._lr_end + 0.5 * (self._lr_start - self._lr_end) * (
                1.0
                + math.cos(
                    math.pi
                    * (step_count - self._lr_warmup_iter)
                    / (self._all_iter - self._lr_warmup_iter - self._lr_saturate_iter)
                )
            )
        return [cur_lr for _ in self.optimizer.param_groups]


SCHEDULER_COLLECTION: Dict[str, Type(_LRScheduler)] = {
    "cosine": CosineSchedulerWarmup
}



#TODO[Ziteng]: warmup wrapper
class WarmupSaturationWrapper(_LRScheduler):
    def __init__(
        self, 
        scheduler: BaseScheduler, 
        lr_warmup_ratio: float = 1.0, 
        warmup_epoch: int = 0, 
        lr_saturation_ratio: float = 1.0, 
        saturation_epoch: int = 0
    ) -> None:
        self._scheduler = scheduler
        self._base_lr_list: List[float] = scheduler.base_lrs
        
        self._lr_warmup_start_list: List[float] = [lr_warmup_ratio * base_lr for base_lr in self._base_lr_list]
        self._warmup_epoch = warmup_epoch
        self._lr_saturation_list: List[float] = [lr_saturation_ratio * base_lr for base_lr in self._base_lr_list]
        self._saturation_epoch = saturation_epoch
        
        self._epoch_num = self._scheduler.epoch_num

    def get_lr(self) -> List[float]:
        
        if self.last_epoch < self._warmup_epoch:
            # warmup learning rate
            lr_list: List[float] = []
            for base_lr, lr_warmup_start in zip(self._base_lr_list, self._lr_warmup_start_list):
                lr_list.append(
                    (base_lr - lr_warmup_start) * self.last_epoch/self._warmup_epoch \
                        + self.lr_warmup_start
                )
        elif self._epoch_num <= self.last_epoch < self._epoch_num + self._saturation_epoch:
            lr_list = self._lr_saturation_list
        else:
            lr_list = self._scheduler.get_lr()
            
        return lr_list
    
    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch: int = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.last_epoch < self._warmup_epoch:
            for param_group, lr in zip(self._scheduler.param_groups, self.get_lr()):
                param_group["lr"] = lr
        elif self._epoch_num <= self.last_epoch < self._epoch_num + self._saturation_epoch:
            for param_group, lr in zip(self._scheduler.param_groups, self.get_lr()):
                param_group["lr"] = lr
        else:
            self._scheduler.step(epoch=self.last_epoch - self._warmup_epoch)
                



