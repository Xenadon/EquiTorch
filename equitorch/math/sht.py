from functools import lru_cache

import torch
from torch import Tensor

from torch_geometric.utils import scatter

import e3nn
from e3nn import o3
from e3nn.o3._s2grid import _quadrature_weights


from ..utils.indices import check_degree_range, degrees_in_range, orders_in_degree_range
from ..typing import DegreeRange

def s2_grid(num_thetas: int, num_phis: int, device = None, dtype = None):
    r"""
    Returns the regular grid points on a sphere as defined by the spherical coordinates theta and phi.

    The grid points are calculated according to the following formulas:

    .. math::
        \begin{aligned}
        \theta_i &= \pi\cdot\frac{(i+\frac{1}{2})}{N}, &i = 0,1,\dots N-1,\\
        \phi_j &= 2\pi\cdot\frac{j}{M}, &j = 0,1,\dots M-1,
        \end{aligned}

    where :math:`N` and :math:`M` are the number of points along the 
    :math:`\theta` and :math:`\phi` axes, respectively.

    Parameters
    ----------
    num_thetas : int
        The number of grid points along the :math:`\theta` axis.
    num_phis : int
        The number of grid points along the :math:`\phi` axis.
    device : optional
        The device to store the resulting tensors on. Defaults is None.
    dtype : torch.dtype, optional
        The data type of the resulting tensors. Defaults is None.

    Returns
    -------
    (Tensor, Tensor)
        A tuple of two tensors of the grid points along the :math:`\theta` and :math:`\phi` axes.
    """
    return o3.s2_grid(num_thetas, num_phis, dtype, device)

@lru_cache(None)
def _isht_prepare(L: DegreeRange, num_thetas: int, num_phis: int, dtype=None, device=None):
    assert num_thetas % 2 == 0
    L = check_degree_range(L)
    thetas, phis = s2_grid(num_thetas, num_phis, device, dtype)
    weighted_legendre = o3.Legendre(
        list(degrees_in_range(L)))(thetas.cos(), thetas.sin().abs()).unsqueeze(-1)
    # [num_thetas, M, 1]
    fourier = o3.spherical_harmonics_alpha(L[1], phis).unsqueeze(0)  # [1, num_phis, m]
    m_in_range = orders_in_degree_range(L, device=device)
    def _isht(X: Tensor):
        '''
        X must be of shape [...,M,C]
        '''
        X_ = X.view(-1,1,X.shape[-2],X.shape[-1])
        inter = X_ * weighted_legendre
        reduced_l = scatter(inter, m_in_range, dim=-2).flatten(0,1) # ([...,theta),m,C]
        return (fourier @ reduced_l).view(*(X.shape[:-2]),num_thetas,num_phis,X.shape[-1])
    return _isht

@lru_cache(None)
def _sht_prepare(L: DegreeRange, num_thetas: int, num_phis: int, dtype=None, device=None):
    assert num_thetas % 2 == 0
    L = check_degree_range(L)
    thetas, phis = s2_grid(num_thetas, num_phis, device, dtype)
    weight = torch.pi * 4 * _quadrature_weights(num_thetas // 2, dtype=dtype, device=device) * num_thetas**2 / num_phis
    weighted_legendre = o3.Legendre(
        list(degrees_in_range(L)))(thetas.cos(), thetas.sin().abs()) * weight.unsqueeze(-1)# [num_thetas, M]
    fourier = o3.spherical_harmonics_alpha(L[1], phis).transpose(-1,-2) # [1, m, num_phis]
    m_in_range = orders_in_degree_range(L, device=device)
    def _sht(x: Tensor):
        '''
        x must be of shape [...,theta,phi,C]
        '''
        # x_ = x.view(-1,x.shape[-2],x.shape[-1])
        inter = (fourier @ x) # ...,theta, m, C
        expand_l = inter.index_select(-2, m_in_range) # ..., theta, M, C
        return torch.einsum('...tMC,tM->...MC', expand_l, weighted_legendre).unflatten(0,x.shape[:-3])
    return _sht


def isht(X: Tensor, L: DegreeRange, num_thetas: int, num_phis: int):
    r'''
    Inverse spherical harmonics transform.

    .. math::
        F(\theta_{i},\phi_j)=\sum_{l\in L}\sum_{m=-l}^l X_{m}^{(l)}Y_{m}^{(l)}(\theta_i,\phi_j),\\
    
    where :math:`\theta_i=\frac{\pi i+\frac{\pi}{2}}{\text{num_thetas}}`, :math:`\phi_j=\frac{2\pi j}{\text{num_phis}}`.

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        Input tensor of shape :math:`(..., \text{num_orders}, C)`.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of spherical harmonics.
    num_thetas : int
        The number of grid points of :math:`\theta`.
    num_phis : int
        The number of grid points of :math:`\phi`.

    Returns
    -------
    :obj:`~torch.Tensor`
        Output tensor of shape :math:`(..., \text{num_thetas}, \text{num_phis}, C)`.
    
    Example
    -------
    >>> L = (1,3)
    >>> C = 3
    >>> num_thetas = 34
    >>> num_phis = 18
    >>> N = 6
    >>> X = Tensor(torch.randn(N,num_orders_in(L),C)).to(device)
    >>> print(X.shape)
    torch.Size([6, 15, 3])
    >>> x = isht(X, L, num_thetas, num_phis)
    >>> print(x.shape)
    torch.Size([6, 34, 18, 3])
    >>> X_ = sht(x, L, num_thetas, num_phis)
    >>> print(X_.shape)
    torch.Size([6, 15, 3])
    >>> print((X-X).abs().max())
    tensor(0.)
    '''

    _isht = _isht_prepare(L, num_thetas, num_phis, device=X.device, dtype=X.dtype)
    return _isht(X)


def sht(x: Tensor, L: DegreeRange, num_thetas: int, num_phis: int):
    r'''
    The spherical harmonics transform of a function :math:`F(\theta, \phi)` is
    defined as:

    .. math::
        X_{m}^{(l)} = \int_{S^2} F(\theta, \phi) Y_{m}^{(l)}(\theta, \phi) \sin\theta \mathrm{d}\phi \mathrm{d}\theta

    approximated by:

    .. math::
        X_{m}^{(l)} \approx \sum_{i=0}^{\text{num_thetas}-1} \sum_{j=0}^{\text{num_phis}-1} F(\theta_i, \phi_j) Y_{m}^{(l)}(\theta_i, \phi_j) \sin\theta_i \Delta\theta \Delta\phi

    where :math:`\theta_i=\frac{\pi i+\frac{\pi}{2}}{\text{num_thetas}}`,
    :math:`\phi_j=\frac{2\pi j}{\text{num_phis}}`,
    :math:`\Delta\theta=\frac{\pi}{\text{num_thetas}}`,
    :math:`\Delta\phi=\frac{2\pi}{\text{num_phis}}`.

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        Input tensor of shape :math:`(..., \text{num_thetas}, \text{num_phis}, C)`.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of spherical harmonics.
    num_thetas : int
        The number of grid points of :math:`\theta`.
    num_phis : int
        The number of grid points of :math:`\phi`.

    Returns
    -------
    :obj:`~torch.Tensor`
        Output tensor of shape :math:`(..., \text{num_orders}, C)`.
    '''

    _sht = _sht_prepare(L, num_thetas, num_phis, device=x.device, dtype=x.dtype)
    return _sht(x)