import torch
from torch import nn

class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results - target)**2

        return d
