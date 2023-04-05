from typing import Any, Optional, Callable
from abc import ABC
from abc import abstractmethod
import torch
import torch.nn as nn
from Models.CommonBlocks.BaseBlocks import ResBlock
from Models.CommonBlocks.BaseBlocks import ResBlockPreAct

class Darknet(nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def _make_stage(
        self, 
        in_channels: int, 
        out_channels: int, 
        block_type: nn.Module, 
        block_num: int, 
        act: callable[[bool], nn.Module] = nn.ReLU
    ) -> nn.Sequential:
        stage = nn.Sequential(
            
        )
        return stage
