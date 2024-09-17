import torch
from torch import Tensor
import torch.nn as nn

class CosineCutoff(nn.Module):
    """
    Implements a cosine cutoff function.

    This cutoff function smoothly decreases from 1 to 0 in the range [0, cutoff]
    using a cosine function.

    :param cutoff: The cutoff distance beyond which the function returns 0.
    :type cutoff: float

    :Example:

    >>> cutoff = CosineCutoff(5.0)
    >>> x = torch.tensor([1.0, 3.0, 5.0, 7.0])
    >>> cutoff(x)
    
    """
    def __init__(self, cutoff: float, start: float = 0):
        super().__init__()        
        assert start < cutoff
        self.cutoff = cutoff
        self.start = start
        self.range = self.cutoff - self.start

    def forward(self, x: Tensor):
        x = (x.clamp(self.start, self.cutoff) - self.start) / self.range
        return 0.5 * (1.+torch.cos(torch.pi * x))
    
class MollifierCutoff(nn.Module):

    def __init__(self, cutoff: float, start: float = 0, eps: float = 1e-9):
        super().__init__()
        assert start < cutoff
        self.cutoff = cutoff
        self.start = start
        self.range = self.cutoff - self.start
        self.eps = eps
    
    def forward(self, x:Tensor):
        x = (x.clamp(self.start, self.cutoff) - self.start) / (self.range + self.eps)
        return torch.exp(1-1/(1-x.pow(2)))

class PolynomialCutoff(nn.Module):

    def __init__(self, cutoff: float, start: float = 0, p: int = 5):
        super().__init__()
        assert start < cutoff
        self.cutoff = cutoff
        self.start = start
        self.range = self.cutoff - self.start
        self.p = p
        self.a = -(self.p+1)*(self.p+2)/2
        self.b = self.p*(self.p+2)
        self.c = -self.p*(self.p+1)/2

    def forward(self, x:Tensor):
        x = (x.clamp(self.start, self.cutoff) - self.start) / self.range
        return 1 + self.a * x.pow(self.p) + self.b * x.pow(self.p+1) + self.c * x.pow(self.p+2)