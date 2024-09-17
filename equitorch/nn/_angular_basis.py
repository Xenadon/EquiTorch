from math import sqrt, pi
from typing import Optional, Union, overload
import torch
import torch.nn as nn
from torch import Tensor

class SineBasisExpansion(torch.nn.Module):
    """
    Sine Basis Expansion module.

    This module implements a sine basis expansion of the form:

    .. math::

        \sin(n x)

    where :math:`n` is the frequency that ranges from :math:`1` to :math:`\text{max_freq}` and :math:`x` is the input.

    :param max_freq: The maximum frequency to use.
    :type max_freq: int

    Attributes:
        max_freq (int): The maximum frequency.
        freq (Tensor): The frequency parameters.

    """
    def __init__(self, 
                 max_freq: int):
        super().__init__()
        self.max_freq = max_freq
        freq = torch.arange(1, max_freq+1)
        self.register_buffer('freq', freq)

    def forward(self, x: torch.Tensor):
        return (self.freq * x.unsqueeze(-1)).sin()
    
class CosineBasisExpansion(torch.nn.Module):
    """
    Cosine Basis Expansion module.

    This module implements a sine basis expansion of the form:

    .. math::

        \cos(n x)

    where :math:`n` is the frequency that ranges from :math:`1` to :math:`\text{max_freq}` and :math:`x` is the input.

    :param max_freq: The maximum frequency to use.
    :type max_freq: int

    Attributes:
        max_freq (int): The maximum frequency.
        freq (Tensor): The frequency parameters.
    """
    def __init__(self, 
                 max_freq: int):
        super().__init__()
        self.max_freq = max_freq
        freq = torch.arange(1, max_freq+1)
        self.register_buffer('freq', freq)

    def forward(self, x: torch.Tensor):
        return (self.freq * x.unsqueeze(-1)).cos()


class FourierBasisExpansion(torch.nn.Module):
    """
    Fourier Basis Expansion module.

    This module implements a Fourier basis expansion of the form:

    .. math::

        [\sin(nx), \cos(nx)]

    where :math:`n` is the frequency that ranges from :math:`1` to :math:`\text{max_freq}` and :math:`x` is the input.
    Optionally, it can include a constant term (frequency 0).

    :param max_freq: The maximum frequency to use.
    :type max_freq: int
    :param include_freq_0: Whether to include a constant term (frequency 0) in the expansion. default=False.
    :type include_freq_0: bool, optional

    Attributes:
        max_freq (int): The maximum frequency.
        freq (Tensor): The frequency parameters.
        include_freq_0 (bool): Whether to include a constant term.

    Note:
        The output tensor is organized in the following frequency order:
        `[sin(max_freq*x), ..., sin(2x), sin(x), 1 (if include_freq_0), cos(x), cos(2x), ..., cos(max_freq*x)]`
        
        This arrangement places lower frequencies closer to the center of the output tensor,
        with sine terms in descending order followed by cosine terms in ascending order.

        It is recommended to not include frequency 0. It will be equivalent to add a bias in 
        the consequential linear operation. 
    """

    def __init__(self, 
                 max_freq: int,
                 include_freq_0: bool = False):
        super().__init__()
        self.max_freq = max_freq
        freq = torch.arange(1, max_freq+1)
        self.register_buffer('freq', freq)
        self.include_freq_0 = include_freq_0

    def forward(self, x: torch.Tensor):
        """
        Compute the Fourier basis expansion of the input.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Fourier basis expansion of the input. The output tensor has shape (..., 2*max_freq) if include_freq_0 is False,
                 or (..., 2*max_freq + 1) if include_freq_0 is True.
        :rtype: torch.Tensor
        """
        x = x.unsqueeze(-1)
        if self.include_freq_0:
            return torch.cat([
                (self.freq.flip(0) * x).sin(),
                torch.ones_like(x),                                                                   
                (self.freq * x).cos()
            ], dim=-1)
        else:
            return torch.cat([
                (self.freq.flip(0) * x).sin(),
                (self.freq * x).cos()
            ], dim=-1)