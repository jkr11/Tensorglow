import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, *args, norm=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)   
        self.kwargs = kwargs
        self.norm = norm

    def forward(self, x):
        F.normalize(self.weight.data, p=self.norm)
        return super().forward(x)