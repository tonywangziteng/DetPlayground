import torch
import torch.nn as nn

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self._inplace = inplace

    def forward(self, x: torch.Tensor):
        if self._inplace:
            return x.mul_(torch.sigmoid(x))
        else:
            return x * torch.sigmoid(x)
