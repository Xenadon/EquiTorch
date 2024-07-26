from math import sqrt, pi
from typing import Optional, Union
import torch
import torch.nn as nn
from torch import Tensor

# class RectBinsExpansion(nn.Module):
#     def __init__(self,
#                  width: Union[Tensor,float], 
#                  num_basis: Optional[float] = None, 
#                  start: Optional[float] = None,
#                  end: Optional[float] = None,
#                  center: Optional[Tensor] = None):
#         if num_basis is None:
#             self.center = center
#         else:
#             self.center = torch.linspace(start, end, num_basis)
#         if isinstance(width, float):
#             self.width = torch.ones_like(self.center) * width
#         self.lb = self.center - self.width / 2
#         self.ub = self.center + self.width / 2

#     def forward(self, x:Tensor):
#         x = x.unsqueeze(-1)
#         return ((x > self.lb) & (x < self.ub)).type_as(x)

class GaussianBasisExpansion(nn.Module):
    '''
    exp(-gamma * (r-mu)^2)
    '''

    def __init__(self, 
                 gamma: Union[Tensor,float], 
                 num_basis: Optional[float] = None, 
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 mu: Optional[Tensor] = None,
                 trainable: bool = False):
        if num_basis is None:
            self.mu = mu
        else:
            self.mu = torch.linspace(start, end, num_basis)
        if isinstance(gamma, float):
            self.gamma = torch.ones_like(self.mu) * gamma

        if trainable:
            self.mu = nn.Parameter(self.mu)
            self.gamma = nn.Parameter(self.gamma)

    def forward(self, x: Tensor):
        return torch.exp(-self.gamma * (x.unsqueeze(-1)-self.mu).pow(2))