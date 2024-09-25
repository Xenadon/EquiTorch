
from math import sqrt

import torch
from torch import Tensor

from torch_geometric.utils import scatter

from e3nn import o3

import functools

from ..utils._indices import check_degree_range, reduce_order_to_degree
from ..typing import DegreeRange
from ..typing import Union, Tuple

from ._o3 import *

def dot(x1: Tensor, x2: Tensor, L: DegreeRange, channel_wise=True):
    r"""
    Compute the degree-wise dot product between spherical features.

    .. math::

        d_c^{(l)} = \sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_c [{\mathbf{y}_m^{(l)}}]_c

    if :obj:`channel_wise`, or

    .. math::

        d_{c_1,c_2}^{(l)} = \sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_{c_1} [{\mathbf{y}_m^{(l)}}]_{c_2}
        
    otherwise.

    Parameters
    ----------
    x1 : Tensor
        First input tensor of shape :math:`(N, \text{num_orders_1}, C_1)`.
    x2 : Tensor
        Second input tensor of shape :math:`(N, \text{num_orders_1}, C_2)`.
    L : DegreeRange
        The degree range of inputs.
    channel_wise : bool, optional
        If True, compute channel-wise dot product. Default is True.

    Returns
    -------
    Tensor
        The result of the dot product of shape 
        :math:`(N, \text{num_degrees}, C)` if :obj:`channel_wise` is :obj:`True`
        or :math:`(N, \text{num_degrees}, C_1, C_2)` if :obj:`channel_wise` is :obj:`False`.

        
    Where :math:`N` is the batch-size that will automatically broadcast if set to :math:`1`
    and :math:`C_1,C_2,C` are corresponding number of channels.
    If :obj:`channel_wise` is :obj:`True`, :math:`C_1=C_2=C` should be satisfied.  

    Examples
    --------
    >>> x1 = torch.randn(32, 16, 64)  # (N, num_orders, C1)
    >>> x2 = torch.randn(32, 16, 64)  # (N, num_orders, C2)
    >>> L = DegreeRange(0, 3)
    >>> result = dot(x1, x2, L)
    >>> print(result.shape)
    torch.Size([32, 4, 64])  # (N, num_degrees, C)

    >>> result_not_channel_wise = dot(x1, x2, L, channel_wise=False)
    >>> print(result_not_channel_wise.shape)
    torch.Size([32, 4, 64, 64])  # (N, num_degrees, C1, C2)
    """

    L = check_degree_range(L)
    if not channel_wise:
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-2)
        return reduce_order_to_degree(x1*x2, L, dim=-3)
    else:
        return reduce_order_to_degree(x1*x2, L, dim=-2)


def norm(x: Tensor, L: DegreeRange):
    r"""Compute the degree & channel-wise norm of a spherical feature as 

    .. math::

        \|\mathbf{x}^{(l)}_c\| = \sqrt{\sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_c^2}, l\in L

    Parameters
    ----------
    x : Tensor
        Spherical feature tensor of shape :math:`(N, \text{num_orders}, C)`
    L : DegreeRange
        The degree range of input

    Returns
    -------
    Tensor
        Norm tensor of shape :math:`(N, \text{num_degrees}, C)`

    
    Where :math:`N` is the batch-size and :math:`C` is the number of channels. 
    """

    return reduce_order_to_degree(x**2, L, dim=-2).sqrt()

def norm2(x: Tensor, L: DegreeRange):
    r"""Compute the square of degree & channel-wise norm of a spherical feature as

    .. math::

        \|\mathbf{x}^{(l)}_c\|^2 = {\sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_c^2}, l\in L

    Parameters
    ----------
    x : Tensor
        Spherical feature tensor of shape :math:`(N, \text{num_orders}, C)`
    L : DegreeRange
        The degree range of input

    Returns
    -------
    Tensor
        Square of norm tensor of shape :math:`(N, \text{num_degrees}, C)`

    
    Where :math:`N` is the batch-size and :math:`C` is the number of channels. 
    """
    return reduce_order_to_degree(x**2, L, dim=-2)

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
    return o3.s2_grid(num_thetas, num_phis, device, dtype)

# def spherical_harmonics_on_grid(grid_points: Union[Tensor, Tuple[Tensor, Tensor]]):
def legendre_fourier_on_grid(L: int, res_theta: float, res_phi: float, device = None, dtype = None):
    """
    Returns the legendre-cos and fourier functions on the regular grid points on a sphere as defined by the spherical coordinates theta and phi.

    Parameters
    ----------
    L : int
        _description_
    res_theta : float
        _description_
    res_phi : float
        _description_
    device : _type_, optional
        _description_, by default None
    dtype : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    theta, phi, legendres, fouriers = o3.spherical_harmonics_s2_grid(
        lmax=L, res_beta=res_theta, res_alpha=res_phi, device=device, dtype=dtype)
    return (theta.unsqueeze(-1), phi.unsqueeze(-2)),  legendres.unsqueeze(-1), fouriers.unsqueeze(-2)

# def s2_spectral_to_spatial(x: Tensor, grid_points: Union[Tensor, Tuple]):

def spherical_harmonics_evaluation(grid_points: Union[Tensor, Tuple[Tensor, Tensor]],
                         coefficients: Tensor,
                         l_max: int,
                         dim_m: int = -1):
    if isinstance(grid_points, Tuple):
        ys = o3.spherical_harmonics_alpha_beta(range(l_max+1), grid_points[1], grid_points[0])
