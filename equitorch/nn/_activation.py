from math import log
from typing import Callable, Optional, Union, Tuple
from ..typing import DegreeRange

import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
from ..utils._indices import check_degree_range, expand_degree_to_order
from ..math import norm

import e3nn.o3 as o3

class NormAct(nn.Module):
    r"""Apply an activation function to the norm of spherical tensor features.

    This module computes the norm of the input tensor along specific dimensions,
    applies an activation function to the norm, and then scales the original input
    by the activated norm.

    Parameters
    ----------
    activation : Callable[[Tensor], Tensor]
        The activation function to apply to the norm.
    L : DegreeRange
        The range of degrees (l_min, l_max) to consider for spherical harmonics.
    degree_wise : bool, optional
        If True, apply the activation separately for each degree. 
        If False, apply it to the overall norm. Default is False.
    bias : bool, optional
        If True, add a learnable bias before the activation. Default is False.
    in_channels : int, optional
        Number of input channels. Required if bias is True.
    eps : float, optional
        A small value added to the norm for numerical stability. Default is 1e-9.

    Attributes
    ----------
    activation : Callable[[Tensor], Tensor]
        The scalar activation function.
    degree_wise : bool
        Whether the activation is applied degree-wise.
    L : DegreeRange
        The range of degrees.
    num_degrees : int
        The number of degrees in the range L.
    need_bias : bool
        Whether bias is used.
    in_channels: Optional[int]
        The number of channels, when need_bias is True.

    Notes
    -----
    - If degree_wise is True, the norm is computed for each degree separately.
    - If degree_wise is False, the norm is computed across all degrees.
    - The bias, if used, is added before the activation function is applied.
    - After activation, the input is scaled by the activated norm.

    Examples
    --------
    >>> activation = torch.nn.SiLU()
    >>> L = (0, 2)  # degrees 0, 1, 2
    >>> norm_act = NormAct(activation, L, degree_wise=True, bias=True, in_channels=16)
    >>> x = torch.randn(32, 9, 16)  # batch_size=32, num_orders=9 (for l=0,1,2), channels=16
    >>> output = norm_act(x)
    >>> print(output.shape)
    torch.Size([32, 9, 16])
    """

    def __init__(self, 
                 activation: Callable[[Tensor], Tensor],
                 L: DegreeRange, 
                 degree_wise:bool=False, 
                 need_bias:bool=False, 
                 in_channels: Optional[int] = None):
        super().__init__()
        self.activation = activation
        self.degree_wise = degree_wise
        self.L = check_degree_range(L)
        self.num_degrees = self.L[1] - self.L[0] + 1
        self.need_bias = need_bias
        if self.need_bias:
            self.num_channels = in_channels
            if self.degree_wise:
                self.bias = nn.Parameter(
                    torch.randn(self.num_degrees, self.num_channels))
            else:
                self.bias = nn.Parameter(
                    torch.randn(self.num_channels))

    def forward(self, x: Tensor):
        if self.degree_wise:
            n = norm(x, self.L)
            if self.need_bias:
                n = n + self.bias
            n = self.activation(n)
            x = x * expand_degree_to_order(n, self.L, dim=-1)
        else:
            n = torch.norm(x, dim=(-2))
            if self.need_bias:
                n = n + self.bias
            n = self.activation(n)
            x = x * n.unsqueeze(-2)
        return x

class GatedAct(nn.Module):
    r"""Applies gated activation to a set of variables.

    This module applies an element-wise gating mechanism, where one set of variables
    (gate variables) controls the activation of another set of variables (acted variables).

    Parameters
    ----------
    num_gate_vars : int
        Number of gate variables.
    num_acted_vars : int
        Number of variables to be acted upon.
    activation : Callable[[Tensor], Tensor]
        Act function to be applied to the gate variables.
    project : bool, optional
        If True, applies a linear projection to gate variables. Default is False.

    Attributes
    ----------
    num_gate_vars : int
        Number of gate variables.
    num_acted_vars : int
        Number of variables to be acted upon.
    activation : Callable[[Tensor], Tensor]
        Act function for gate variables.
    project : bool
        Whether to project input scalars to gating scalars.

    Notes
    -----
    If `project` is False, `num_gate_vars` must be equal to `num_acted_vars`.
    """
    def __init__(self, 
                 num_gate_vars: int, 
                 num_acted_vars: int, 
                 activation: Callable[[Tensor], Tensor], 
                 project: bool = False):
        super().__init__()
        if project:
            self.project = nn.Linear(num_gate_vars, num_acted_vars, bias=False)
        else:
            assert num_gate_vars == num_acted_vars
            self.project = None
        self.num_gate_vars = num_gate_vars
        self.num_acted_vars = num_acted_vars
        self.activation = activation

    def forward(self, gate_vars, acted_vars, dim=-1):
        r"""Apply gated activation to the input tensors.

        Parameters
        ----------
        gate_vars : Tensor
            Gate variables tensor of shape (..., num_gate_vars).
        acted_vars : Tensor
            Variables to be acted upon, tensor of shape (..., m, num_acted_vars),
            where m is an arbitrary number of dimensions.
        dim : int, optional
            The dimension along which to apply the gating mechanism. Default is -1.

        Returns
        -------
        Tensor
            Output tensor after applying gated activation, with the same shape as `acted_vars`.
        """
        if self.project:
            gate_vars = self.project(gate_vars)
        self.activation(gate_vars.unsqueeze(dim)) * acted_vars

class S2Act(nn.Module):

    def __init__(self, num_resolution:Union[Tuple[int,int],int], 
                 activation: Callable[[Tensor], Tensor],
                 L: DegreeRange):
        self.L_max = check_degree_range(L)[1]
        if isinstance(num_resolution, int):
            self.num_resolution = (num_resolution, num_resolution)
        else:
            self.num_resolution = num_resolution

        self.L = L

        self.activation = activation
        self.to_grid = o3.ToS2Grid(self.L_max, *self.num_resolution)
        self.from_grid = o3.FromS2Grid(self.L_max, *self.num_resolution)

    def forward(self, x: Tensor):
        return self.from_grid(self.activation(self.to_grid(x)))
        

class SepS2Act(nn.Module):
    
    def __init__(self, num_resolution:Union[Tuple[int,int],int], 
                 activation: Callable[[Tensor], Tensor],
                 L: DegreeRange):
        self.L_max = check_degree_range(L)[1]
        if isinstance(num_resolution, int):
            self.num_resolution = (num_resolution, num_resolution)
        else:
            self.num_resolution = num_resolution

        self.L = L

        self.activation = activation
        self.to_grid = o3.ToS2Grid(self.L_max, *self.num_resolution)
        self.from_grid = o3.FromS2Grid(self.L_max, *self.num_resolution)

    def forward(self, x: Tensor):
        return self.from_grid(self.activation(self.to_grid(x)))


def shifted_softplus(x: Tensor, alpha: float = 0.5, threshold: float = 20.):
    r"""Compute the shifted softplus activation function.

    This function implements a modified version of the softplus activation:

    .. math::

        \text{ssp}(x) = \begin{cases}
            \ln(\alpha \exp(x) + (1-\alpha)) & \text{if } x < \text{threshold} \\
            x + \ln(\alpha) & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float, optional
        Scaling factor in the exponential term. Default is 0.5.
    threshold : float, optional
        Threshold value for numerical stability. Default is 20.0.

    Returns
    -------
    Tensor
        Output tensor after applying the shifted softplus activation.

    Notes
    -----
    - This function provides a smooth approximation to the ReLU function.
    - The threshold is used to prevent numerical overflow for large input values.
    - When x is large (>= threshold), the function approximates to a linear function.
    """
    return torch.where(x < threshold, 
                       torch.log(alpha*(torch.exp(x)) + (1-alpha)),
                       x + log(alpha))


class ShiftedSoftPlus(nn.Module):
    r'''
    See :obj:`shifted_softplus`
    '''
    def __init__(self, alpha: float = 0.5, threshold: float = 20.):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, x: Tensor):
        return shifted_softplus(x, self.alpha, self.threshold)