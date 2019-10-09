import torch
from torch import nn
from torch.nn import init


class TransE(nn.Module):
    """TransE Model"""
    def __init__(self):
        super(TransE, self).__init__()

    def forward(self, x):
        
        return x