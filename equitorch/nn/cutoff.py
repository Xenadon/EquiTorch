import torch
from torch import Tensor
import torch.nn as nn

class CosineCutoff(nn.Module):
    r"""
    The cosine cutoff function

    .. math::
        f(r)=\begin{cases}
        1, & r < \text{start},\\
        \dfrac{1}{2}\left[1 + \cos\left(\pi\cdot u\right)\right], & \text{start}\le r \le \text{cutoff},\\
        0, & r > \text{cutoff},
        \end{cases}

    where :math:`u=\dfrac{r-\text{start}}{\text{cutoff}-\text{start}}`.

    This cutoff function smoothly decreases from 1 to 0 in the range 
    :math:`[\text{start}, \text{cutoff}]` using a cosine function.

    Parameters
    ----------
    cutoff : float
        The cutoff distance where the function reaches zero.
    start : float, optional
        The starting distance where the function begins to decrease from 1.
        Must be less than `cutoff`. Default is 0.
    """
    def __init__(self, cutoff: float, start: float = 0):
        super().__init__()        
        assert start < cutoff
        self.cutoff = cutoff
        self.start = start
        self.range = self.cutoff - self.start

    def forward(self, x: Tensor):
        """
        """
        x = (x.clamp(self.start, self.cutoff) - self.start) / self.range
        return 0.5 * (1.+torch.cos(torch.pi * x))
    
class MollifierCutoff(nn.Module):
    r'''
    The mollifier cutoff function

    .. math::
        f(r) = \begin{cases} 
        1, & r < \text{start}\\
        \exp \left[{1 - \left({1 - u^2}+\epsilon\right)^{-1}}\right] & \text{start} \le r \le \text{cutoff} \\
        0, & r > \text{cutoff}  ,
        \end{cases}
        
    where :math:`u=\dfrac{r-\text{start}}{\text{cutoff}-\text{start}}`.

    Parameters
    ----------
    cutoff : float
        The cutoff distance where the function reaches zero.
    start : float, optional
        The starting distance where the function begins to decrease from 1.
        Must be less than `cutoff`. Default is 0.
    eps : float, optional
        A small value to prevent division by zero or numerical instabilities.
        Default is 1e-9.
    '''
    def __init__(self, cutoff: float, start: float = 0, eps: float = 1e-9):
        super().__init__()
        assert start < cutoff
        self.cutoff = cutoff
        self.start = start
        self.range = self.cutoff - self.start
        self.eps = eps
    
    def forward(self, x:Tensor):
        """
        """
        x = (x.clamp(self.start, self.cutoff) - self.start) / (self.range)
        return torch.exp(1-1/(1-x.pow(2) + self.eps))

class PolynomialCutoff(nn.Module):
    r'''
    Implements a polynomial cutoff function

    .. math::
        f(r) = \begin{cases} 
        1, & r < \text{start},\\
        1-\dfrac{(p+1)(p+2)}{2}u^p+p(p+2)u^{p+1}-\frac{p(p+1)}{2}u^{p+2}& \text{start} \le r \le \text{cutoff}, \\
        0, & r > \text{cutoff},  
        \end{cases}

    where :math:`u=\dfrac{r-\text{start}}{\text{cutoff}-\text{start}}`.

    Parameters
    ----------
    cutoff : float
        The cutoff distance where the function reaches zero.
    start : float, optional
        The starting distance where the function begins to decrease from 1.
        Must be less than `cutoff`. Default is 0.
    p : int, optional
        The order of the polynomial.
        Default is 5.
    '''
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
        """
        """
        x = (x.clamp(self.start, self.cutoff) - self.start) / self.range
        return 1 + self.a * x.pow(self.p) + self.b * x.pow(self.p+1) + self.c * x.pow(self.p+2)