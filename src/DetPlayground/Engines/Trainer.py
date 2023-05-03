from typing import Dict, Optional, Callable
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from DetPlayground.Dataset.CocoDataset import CocoDataset
from DetPlayground.Loss import LOSS_CALCULATOR_COLLECTION
from DetPlayground.Models import MODEL_COLLECTION
from DetPlayground.Utils.Optimizers import OPTIMIZER_COLLECTION
from DetPlayground.Utils.Schedulers import SCHEDULER_COLLECTION
from DetPlayground.Utils.Schedulers import get_scheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        args: Dict
    ) -> None:
        self._init_model(args["model"])
        self._init_loss_calculator(args["model"])
        
        self._train_dataloader = self._get_dataloader(args["training"]["data"])
        self._val_dataloader = self._get_dataloader(args["validation"]["data"])
        
        self._optimizer = self._get_optimizer(self.model, args["training"]["optimizer"])
    
    def _get_dataloader(
        self, 
        args: Dict,
        data_transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,                    
    ) -> DataLoader:
        dataset = CocoDataset(
            args=args,
            data_transform=data_transform,
            target_transform=target_transform
        )
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=args["batchSize"], 
        )
        return dataloader
        
    def _get_optimizer(self, model: nn.Module, args: Dict) -> optim.Optimizer:
        assert args["type"] in OPTIMIZER_COLLECTION.keys(), \
            "optimizer should be in {}, but got {}".format(OPTIMIZER_COLLECTION.keys(), args["type"])
        optimizer_type = OPTIMIZER_COLLECTION[args["type"]]
        return optimizer_type(model, **args)
    
    def _get_scheduler(self, args: Dict) -> optim.lr_scheduler.LRScheduler:
        assert args["scheduler"] in SCHEDULER_COLLECTION.keys(), \
            "scheduler should be in {}, but got {}".format(SCHEDULER_COLLECTION.keys(), args["scheduler"]) 
        
    def _init_model(
        self, 
        args: Dict
    ):
        model_type = MODEL_COLLECTION[args["type"]]
        self._model = model_type(args)

    def _init_loss_calculator(
        self, 
        args: Dict
    ):
        loss_calculator_type = LOSS_CALCULATOR_COLLECTION[args["type"]]
        self._loss_calculator = loss_calculator_type(**args)

    @property 
    def model(self) -> nn.Module:
        return self._model
    
