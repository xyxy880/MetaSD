import torch
import torch.nn as nn

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))
