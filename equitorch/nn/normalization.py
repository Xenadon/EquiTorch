from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


from ..typing import DegreeRange
from ..math import rms
from ..utils import expand_degree_to_order, degrees_in_range, check_degree_range

class EquivariantLayerNorm(nn.Module):
    r"""
    The equivariant layer RMS normalization.

    If :obj:`degree_wise`, it transforms the input as

    .. math::
        \begin{aligned}
        &\mathbf{x'}^{(l)}_c=\frac{\mathbf{x}^{(l)}_c}{\text{RMS}(\mathbf{x}^{(l)})}\cdot \gamma_c^{(l)},\ l\in L\\
        &\text{RMS}(\mathbf{x}^{(l)})=\sqrt{\frac{1}{C}\sum_c\frac{1}{\text{scale}_l}\sum_{m=-l}^l\big(\mathbf{x}_{m,c}^{(l)}\big)^2}
        \end{aligned}
        
    otherwise, it transforms the input as 

    .. math::
        \begin{aligned}
        &\mathbf{x'}_c=\frac{\mathbf{x}_c}{\text{RMS}(\mathbf{x})}\cdot \gamma_c\\
        &\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{|L|}\sum_{l\in L} \text{RMS}^2(\mathbf{x}^{(l)})}
        \end{aligned}

    where :math:`\gamma`'s are learnable parameters if :obj:`elementwise_affine` and 
    :math:`\text{scale}_l` is :math:`2l+1` if :obj:`degree_scale` or :math:`1` otherwise.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The input degree range.
    channels : int, optional
        The input channels, need to specify if :obj:`elementwise_affine`.
    elementwise_affine : bool, optional
        Whether to have learnable per-element affine parameters 
        initialized to ones. Default is :obj:`True`
    degree_scale : bool, optional
        Whether to scale the norm of each degrees. Default is :obj:`True`
    degree_wise : bool, optional
        Whether the RMS is computed degree-wisely. Default is :obj:`True`
    eps : int, optional
        A value added to the denominator for numerical stability. Default is :obj:`1e-5`

    Example
    -------
    >>> N, C, L = 3, 4, (1,3)   
    >>> x = torch.randn(N,num_orders_in(L),C)
    >>> print(rms(x, L, degree_scale=True, degree_wise=True, keepdim=False))
    tensor([[0.9188, 1.2324, 1.1481],
            [0.8945, 1.0318, 1.0453]])
    >>> norm = EquivariantLayerNorm(L, 4, degree_scale=True, degree_wise=True)
    >>> x1 = norm(x)
    >>> print(rms(x1, L, degree_scale=True, degree_wise=True, keepdim=False))
    tensor([[1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000]], grad_fn=<SqrtBackward0>)
    """
    def __init__(self, 
                 L: DegreeRange, 
                 channels: Optional[int] = None,
                 elementwise_affine: bool =True,
                 degree_scale: bool = True,
                 degree_wise: bool = False,
                 eps: int = 1e-5,
                 ):
        super().__init__()
        self.L = check_degree_range(L)
        self.degree_scale = degree_scale
        self.degree_wise = degree_wise
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.channels = channels

        if self.degree_scale:
            self.register_buffer('scale', (torch.tensor(degrees_in_range(L)).unsqueeze(-1)*2+1))
        else:
            self.scale=None
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(1 if not degree_wise else L[1]-L[0]+1, channels))

    def forward(self, x: Tensor):
        """
        Parameters
        ----------
        x : :obj:`~torch.Tensor`
            Input tensor of shape :math:`(...,\text{num_orders},C)`.

        Returns
        -------
        :obj:`~torch.Tensor`
            Normalized tensor with exactly the same shape of the input.
        """
        sigma2 = rms(x, self.L, 
                    degree_scale=self.degree_scale,
                    degree_wise=self.degree_wise,
                    scale=self.scale,
                    squared=True) + self.eps
        if self.degree_wise:
            sigma2 = expand_degree_to_order(sigma2, self.L, -2)
        x = x * sigma2.rsqrt()
        if self.elementwise_affine:
            weight = self.weight if not self.degree_wise else expand_degree_to_order(self.weight, self.L, -2)
            x = x * weight
        return x