from typing import Optional, Union
import torch
import torch.nn as nn
from torch import Tensor

class GaussianBasisExpansion(nn.Module):
    """
    Gaussian Basis Expansion module.

    This module implements a Gaussian basis expansion of the form:

    .. math::

        \exp(-\gamma_k \cdot (r-\mu_k)^2)

    :param gamma: The gamma parameter for the Gaussian basis function.
    :type gamma: Union[Tensor, float]
    :param num_basis: The number of basis functions to use. If None, mu must be provided.
    :type num_basis: Optional[float], default=None
    :param start: The start value for generating mu values. Required if num_basis is provided.
    :type start: Optional[float], default=None
    :param end: The end value for generating mu values. Required if num_basis is provided.
    :type end: Optional[float], default=None
    :param mu: The mu values for the Gaussian basis functions. Required if num_basis is None.
    :type mu: Optional[Tensor], default=None
    :param trainable: Whether the mu and gamma parameters should be trainable.
    :type trainable: bool, default=False

    Attributes:
        mu (Tensor): The mu values for the Gaussian basis functions.
        gamma (Tensor): The gamma values for the Gaussian basis functions.
        num_basis (int): The number of basis functions.

    .. note::
        If num_basis is provided, mu values are generated using torch.linspace(start, end, num_basis).
        If gamma is a float, it is expanded to match the shape of mu.
        If trainable is True, mu and gamma become nn.Parameter objects.

    """
    def __init__(self, 
                 gamma: Union[Tensor,float], 
                 num_basis: Optional[float] = None, 
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 mu: Optional[Tensor] = None,
                 trainable: bool = False):
        super().__init__()
        if num_basis is None:
            mu = mu
        else:
            mu = torch.linspace(start, end, num_basis)
        if isinstance(gamma, float) or isinstance(gamma, int):
            gamma = torch.ones_like(mu) * gamma

        if trainable:
            self.mu = nn.Parameter(self.mu)
            self.gamma = nn.Parameter(self.gamma)
        else:
            self.register_buffer('mu', mu)
            self.register_buffer('gamma', gamma)
    def forward(self, x: Tensor):

        """
        Compute the Gaussian basis expansion for the input tensor.

        :param x: The input tensor.
        :type x: Tensor
        :return: The Gaussian basis expansion of the input tensor.
        :rtype: Tensor
        """
        return torch.exp(-self.gamma * (x.unsqueeze(-1)-self.mu).pow(2))
    
# Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html#DimeNet
class BesselBasisExpansion(torch.nn.Module):
    """
    Bessel Basis Expansion module.

    This module implements a Bessel basis expansion of the form:

    .. math::

        \frac{\sin(\omega_k (\hat r + \epsilon))}{\hat r + \epsilon},\ k = 1,2,\dots,\text{num_basis}.

    where :math:`\omega_k` is the frequency, default to be :math:`k\pi`, 
    :math:`\hat r=r/c` is the cutoff-normalized distance, 
    and :math:`\epsilon` is a small value for stability near zero.

    :param num_basis: The number of basis functions to use.
    :type num_basis: int
    :param cutoff: The cutoff value for the distance, default=1.
    :type cutoff: float
    :param trainable: Whether the frequency parameters should be trainable.
    :type trainable: bool, default=False
    :param eps: A small value to for stability near zero.
    :type eps: float, default=1e-6

    Attributes:
        num_basis (int): The number of basis functions.
        cutoff (float): The cutoff value for the distance.
        trainable (bool): Whether the frequency parameters are trainable.
        eps (float): A small value to avoid division by zero.
        freq (Tensor): The frequency parameters.

    .. note::
        If trainable is True, the frequency parameters become nn.Parameter objects.

    """
    def __init__(self, 
                 num_basis: int, 
                 cutoff: float = 1,
                 trainable: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.trainable = trainable

        self.eps = eps
        freq = torch.arange(1, num_basis + 1) * torch.pi

        if trainable:
            self.freq = torch.nn.Parameter(freq)
        else:
            self.register_buffer('freq', freq)


    def forward(self, dist: Tensor) -> Tensor:
        """
        Compute the Bessel basis expansion for the input distance tensor.

        :param dist: The input distance tensor.
        :type dist: Tensor
        :return: The Bessel basis expansion of the input distance tensor.
        :rtype: Tensor

        """
        dist = dist.unsqueeze(-1) / self.cutoff + self.eps
        return (self.freq * dist).sin() / (dist)