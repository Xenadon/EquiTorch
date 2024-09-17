
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


# def dot(x1: Tensor, x2: Tensor, L: DegreeRange, channel_wise=True):
#     r"""Compute the degree & channel-wise dot product between spherical features.

#     For spherical features :math:`x_1 = [{x_1}_m^l]_c` and :math:`x_2 = [{x_2}_m^l]_c`,
#     this function computes:

#     .. math::

#         d_c^l = \sum_{m=-l}^l [{x_1}_m^l {x_2}_m^l]_c

#     Parameters
#     ----------
#     x1 : Tensor
#         Spherical feature tensor of shape (N, num_orders, C)
#     x2 : Tensor
#         Spherical feature tensor of shape (N, num_orders, C)
#     L : DegreeRange
#         Range of degrees to consider

#     Returns
#     -------
#     Tensor
#         Dot product tensor of shape (N, num_degrees, C)

#     Notes
#     -----
#     - N is the batch size
#     - num_orders is the total number of spherical harmonic orders
#     - num_degrees is the number of degrees in the specified range L
#     - C is the number of channels
#     """

#     L = check_degree_range(L)
#     return reduce_order_to_degree(x1*x2, L, dim=-2)

def dot(x1: Tensor, x2: Tensor, L: DegreeRange, channel_wise=True):
    """
    Compute the degree-wise dot product between spherical features.

    For spherical features :math:`x_1 = [{x_1}_m^l]_{c1}` and :math:`x_2 = [{x_2}_m^l]_{c2}`,
    this function computes:

    .. math::

        d_c^l = \sum_{m=-l}^l {{x_1}_m^l}_c {{x_2}_m^l}_c

    if channel_wise, or

    .. math::

        d_{c_1,c_2}^l = \sum_{m=-l}^l {{x_1}_m^l}_{c1} {{x_2}_m^l}_{c2}
        
    if not channel_wise.

    Parameters
    ----------
    x1 : Tensor
        Spherical feature tensor of shape (N, num_orders, C1)
    x2 : Tensor
        Spherical feature tensor of shape (N, num_orders, C2)
    L : DegreeRange
        Range of degrees to consider
    channel_wise : bool, optional
        If True, compute channel-wise dot product. Default is True.

    Returns
    -------
    Tensor
        If channel_wise:
            Dot product tensor of shape (N, num_degrees, C)
        If not channel_wise:
            Dot product tensor of shape (N, num_degrees, C1, C2)

    Notes
    -----
    - N is the batch size
    - num_orders is the total number of spherical harmonic orders
    - num_degrees is the number of degrees in the specified range L
    - C, C1, C2 are the number of channels

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
    r"""Compute the degree & channel-wise norm of a spherical feature.

    For a spherical feature :math:`x = [x_m^l]_c`, this function computes:

    .. math::

        \sqrt{\sum_{m=-l}^l [{x_m^l}^2]_c}

    Parameters
    ----------
    x : Tensor
        Spherical feature tensor of shape (N, num_orders, C)
    L : DegreeRange
        Range of degrees to consider

    Returns
    -------
    Tensor
        Norm tensor of shape (N, num_degrees, C)

    Notes
    -----
    - N is the batch size
    - num_orders is the total number of spherical harmonic orders
    - num_degrees is the number of degrees in the specified range L
    - C is the number of channels
    """

    return reduce_order_to_degree(x**2, L, dim=-2).sqrt(2)

def norm2(x: Tensor, L: DegreeRange):
    r"""Compute the square of the degree & channel-wise norm of a spherical feature.

    For a spherical feature :math:`x = [x_m^l]_c`, this function computes:

    .. math::

        \sum_{m=-l}^l [{x_m^l}^2]_c

    Parameters
    ----------
    x : Tensor
        Spherical feature tensor of shape (N, num_orders, C)
    L : DegreeRange
        Range of degrees to consider

    Returns
    -------
    Tensor
        Squared norm tensor of shape (N, num_degrees, C)

    Notes
    -----
    - N is the batch size
    - num_orders is the total number of spherical harmonic orders
    - num_degrees is the number of degrees in the specified range L
    - C is the number of channels
    - This function is equivalent to `norm(x, L)**2` but may be more efficient
    """
    return reduce_order_to_degree(x**2, L, dim=-2)


@functools.lru_cache(maxsize=None)
def s2_grid(num_thetas: int, num_phis: int, device = None, dtype = None):
    return o3.s2_grid(num_thetas, num_phis, device, dtype)

# def spherical_harmonics_on_grid(grid_points: Union[Tensor, Tuple[Tensor, Tensor]]):
def spherical_harmonics_on_grid(L: int, res_theta: float, res_phi: float):
    theta, phi, legendres, fouriers = o3.spherical_harmonics_s2_grid(lmax=3, res_beta=res_theta, res_alpha=res_phi)
    return torch.meshgrid(theta, phi),  legendres.unsqueeze(-1), fouriers.unsqueeze(-2)

# def s2_spectral_to_spatial(x: Tensor, grid_points: Union[Tensor, Tuple]):

def spherical_harmonics_evaluation(grid_points: Union[Tensor, Tuple[Tensor, Tensor]],
                         coefficients: Tensor,
                         l_max: int,
                         dim_m: int = -1):
    if isinstance(grid_points, Tuple):
        ys = o3.spherical_harmonics_alpha_beta(range(l_max+1), grid_points[1], grid_points[0])
