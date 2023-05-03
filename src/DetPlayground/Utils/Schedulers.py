from typing import Dict, Optional, Type
import warnings
import math

from abc import ABC
from abc import abstractmethod

from DetPlayground.Utils.Optimizers import BaseOptimizer

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class BaseScheduler(ABC, LRScheduler):
    """
    The base class for every scheduler
    There are a warmup and a saturation mechanism built in
    """
    def __init__(
        self, 
        optimizer: BaseOptimizer, 
        epoch_num: int, 
        last_epoch: int = -1,
        verbose: bool = False,
        **kargs
    ) -> None:
        LRScheduler.__init__(self, optimizer, last_epoch=last_epoch, verbose=verbose)
        self._epoch_num = epoch_num
        

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


SCHEDULER_COLLECTION: Dict[str, Type(LRScheduler)] = {
    "cosine": CosineSchedulerWarmup
}

#TODO[Ziteng]: warmup wrapper
