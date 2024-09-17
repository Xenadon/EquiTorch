from math import sqrt
from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import scatter, segment

from ..math import dot


from ..typing import DegreeRange

from ..utils._indices import (
    degree_order_to_index,
    expand_degree_to_order,
    check_degree_range,
    extract_batch_ptr,
    extract_in_degree,
    num_order_between,
    degrees_in_range
)
from ..utils._clebsch_gordan import coo_CG

class DegreeWiseLinear(nn.Module):
    '''DegreeWiseLinear.
    '''

    def __init__(self, 
                 L_in: DegreeRange,
                 L_out: DegreeRange,
                 in_channels: int, 
                 out_channels: int,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        assert self.L_out[0] >= self.L_in[0] and self.L_out[1] <= self.L_in[1],\
            f"L_out should be contained in L_in, got L_out={L_out}, L_in={L_in}"
        self.weight = self.self_interaction_weight = nn.Parameter(
                    torch.randn(self.L_out[1]+1 - self.L_out[0], in_channels, out_channels) * 2 / sqrt(in_channels + out_channels))

    def forward(self, x: Tensor):
        x = extract_in_degree(x, self.L_in, self.L_out)
        return (x.unsqueeze(-2) @ 
                expand_degree_to_order(self.weight.unsqueeze(0), self.L_out, dim=-3)).squeeze(-2)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(L_in={self.L_in}, L_out={self.L_out}, in_channels={self.in_channels}, out_channels={self.out_channels})'


def _so2_indices(L_in: DegreeRange, L_out: DegreeRange):
    L_in = check_degree_range(L_in)
    L_out = check_degree_range(L_out)
    ret = sorted([
        (degree_order_to_index(l_out, m_out, L_out[0]), 
         degree_order_to_index(l_in, m_in, L_in[0]),
         l_in,
         l_out,
         -1. if m_out > 0 and m_in < 0 else 1.,
         weight_idx,
         )
        # (weight_idx)
        for weight_idx, (l_out, l_in, m_weight) in enumerate(
            (l_out, l_in, m_weight) 
            for l_out in degrees_in_range(L_out)
            for l_in in degrees_in_range(L_in)
            for m_weight in range(-min(l_out, l_in), min(l_out, l_in)+1)
        ) 
        for m_out, m_in in (((-abs(m_weight),-m_weight), (abs(m_weight),m_weight)) if m_weight != 0 else ((0,0),))
    ])
    Ms = torch.tensor([[t[0], t[1]] for t in ret]).T
    ls = torch.tensor([[t[2], t[3]] for t in ret]).T
    weight_sign = torch.tensor([t[4] for t in ret])
    weight_idx = torch.tensor([t[5] for t in ret])
    return Ms, ls, weight_sign, weight_idx

class SO2Linear(nn.Module):

    def __init__(self, 
                 L_in: DegreeRange, 
                 L_out: DegreeRange, 
                 in_channels: int,
                 out_channels: int,
                 external_weight: bool = False,
                 channel_wise: bool = True
                 ):
        assert in_channels == out_channels or not channel_wise
        super().__init__()
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        self.in_channels = in_channels
        self.out_ms = num_order_between(*self.L_out)
        self.out_channels = out_channels 
        self.channel_wise = channel_wise

        Ms, ls, weight_sign, weight_index = _so2_indices(L_in, L_out)
        
        self.register_buffer('M_out', Ms[0])
        self.register_buffer('M_in', Ms[1])
        self.register_buffer('weight_sign', weight_sign)
        self.register_buffer('weight_index', weight_index)

        self.num_weights = weight_index.max().item()+1

        self.external_weight = external_weight
        
        if self.channel_wise:
            if not self.external_weight:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.channels))
        else:
            if not self.external_weight:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.in_channels, 
                                self.out_channels)
                        * 2 / sqrt(in_channels + out_channels))

    def forward(self, x: Tensor, weight: Optional[Tensor]=None):
        if weight is None and not self.external_weight:
            weight = self.weight
        
        X = x.index_select(dim=1,index=self.M_in) # N * Num * Ci 

        if self.channel_wise:
            W = weight.index_select(dim=1, index=self.weight_index) * self.weight_sign.view(1,-1,1) # N * Num * Ci (* Co)
            out = X * W
        else:
            W = weight.index_select(dim=1, index=self.weight_index) * self.weight_sign.view(1,-1,1,1) # N * Num * Ci (* Co)
            out = (X.unsqueeze(-2)@W).squeeze(-2)
        return scatter(out, index=self.M_out, dim=1, dim_size=self.out_ms)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(L_in={self.L_in}, L_out={self.L_out}, in_channels={self.in_channels},\n  out_channels={self.out_channels}, channel_wise={self.channel_wise}, external_weight={self.external_weight}\n)'
    
def _so3_conv(feature: Tensor, 
              edge_feat: Tensor,
              weight: Tensor,
              CG_vals: Tensor,
              Ml1l2_ptr: Tensor, 
              l_ind: Tensor, 
              M1: Tensor,
              M2: Tensor,
              M_ptr: Tensor, 
              ):
    '''
        feature: B * M1 * C_i
        weight: B * ll1l2 * C_i * C_o
    '''
    feature = feature.index_select(-2, M1) # B * MM1M2 * C
    edge_feat = edge_feat.index_select(-1, M2) # B * MM1M2
    # C_val: MM1M2
    inter = (CG_vals.unsqueeze(0) * edge_feat).unsqueeze(-1) * feature # B * MM1M2 * C
    inter = segment(inter, Ml1l2_ptr.unsqueeze(0))  # B * Ml1l2 * C
    weight = weight.index_select(dim=1,index=l_ind) # B * Ml1l2 * C (* C')

    ret = (inter.unsqueeze(-2) @ weight).squeeze(-2) # B * Ml1l2 * C'

    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C'
    return ret

def _so3_cw_conv(feature: Tensor, 
              edge_feat: Tensor,
              weight: Tensor,
              CG_vals: Tensor,
              M_ptr: Tensor, 
              l_ind: Tensor, 
              M1: Tensor,
              M2: Tensor,
              ):
    '''
        feature: B * M1 * C
        weight: B * ll1l2 * C
    '''
    feature = feature.index_select(-2, M1) # B * MM1M2 * C
    edge_feat = edge_feat.index_select(-1, M2) # B * MM1M2
    # C_val: MM1M2
    inter = (CG_vals.unsqueeze(0) * edge_feat).unsqueeze(-1) * feature # B * MM1M2 * C
    weight = weight.index_select(dim=1,index=l_ind) # B * MM1M2 * C
    ret = inter * weight # B * MM1M2 * C'
    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C
    return ret


class SO3Linear(nn.Module):

    def __init__(
            self,
            L_in: DegreeRange,
            L_edge: DegreeRange,
            L_out: DegreeRange,
            in_channels: int,
            out_channels: int,
            external_weight: bool = False,
            channel_wise: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        assert in_channels == out_channels or not channel_wise

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        self.L_edge = check_degree_range(L_edge)
        self.external_weight = external_weight
        
        self.channel_wise = channel_wise

        CG_vals, Ms, ls_cg, l_ind_cg = coo_CG(L_out, L_in, L_edge)
        
        if self.channel_wise:
            M_batch, M_ptr, M = extract_batch_ptr([Ms[0]])

            l_ind = l_ind_cg

            self.register_buffer('Ms', Ms)
            self.register_buffer('M1', Ms[1])
            self.register_buffer('M2', Ms[2])
            self.register_buffer('l_ind', l_ind)
            self.register_buffer('M_ptr', M_ptr)
            self.register_buffer('CG_vals', CG_vals)

            self.num_weights = l_ind.unique().numel()

            if not self.external_weight:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.in_channels))
        else:
            Ml1l2_batch, Ml1l2_ptr, Ml1l2 = extract_batch_ptr([Ms[0], ls_cg[1], ls_cg[2]])
            
            l_ind = l_ind_cg[Ml1l2_ptr[:-1]]
            M_batch, M_ptr, M = extract_batch_ptr(Ml1l2[:1])

            self.register_buffer('Ml1l2_ptr', Ml1l2_ptr)
            self.register_buffer('l_ind', l_ind)
            self.register_buffer('M1', Ms[1])
            self.register_buffer('M2', Ms[2])
            self.register_buffer('M_ptr', M_ptr)
            self.register_buffer('CG_vals', CG_vals)

            self.num_weights = l_ind.unique().numel()

            if not self.external_weight:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                in_channels, out_channels) \
                                    * sqrt(2/(in_channels+out_channels)))
            
    def forward(self, 
                feature: Tensor, 
                edge_feat: Tensor,
                weight: Optional[Tensor] = None,
        ):
        if weight is None and not self.external_weight:
            weight = self.weight

        if self.channel_wise:
            return _so3_cw_conv(feature, edge_feat, weight,
                         self.CG_vals, self.M_ptr, self.l_ind, 
                         self.M1, self.M2)
        else:
            return _so3_conv(feature, edge_feat, weight,
                            self.CG_vals, self.Ml1l2_ptr, self.l_ind, 
                            self.M1, self.M2, self.M_ptr)
        
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n  L_in={self.L_in}, L_edge={self.L_edge}, L_out={self.L_out},\n  in_channels={self.in_channels}, out_channels={self.out_channels}, channel_wise={self.channel_wise}, external_weight={self.external_weight}\n)'

class ElementWiseLinear(nn.Module):
    r"""Applies an element-wise linear transformation to the input tensor.

    The transformation can be either channel-wise or not channel-wise.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channel_wise (bool, optional): If True, the transformation is channel-wise. Defaults to False.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channel_wise (bool): Indicates whether the transformation is channel-wise.
        weight_shape (tuple): Shape of the weight tensor.

    Returns:
        Tensor: The transformed input tensor.

        If `channel_wise` is False:
            ::math::
                Z_{nmc'} = \sum_{c} X_{nmc} W_{ncc'}

        If `channel_wise` is True:
            ::math::
                Z_{nmc} = X_{nmc} W_{nc} (\text{work like a gate operation})

    """

    def __init__(self, in_channels: int, out_channels: int, 
                 channel_wise = False):
        super().__init__()
        assert in_channels == out_channels or not channel_wise
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_wise = channel_wise
        self.weight_shape = (-1, 1, in_channels, out_channels) if not channel_wise else (-1, 1, in_channels)
    
    def forward(self, x: Tensor, weight: Tensor):
        r"""Applies the element-wise linear transformation to the input tensor.

        Args:
            x (Tensor): Input tensor of shape `(N, num_orders, in_channels)`.
            weight (Tensor): Weight tensor of shape `(N, in_channels)` if `channel_wise` is False, or `(N, in_channels, out_channels)` if `channel_wise` is True.

        Returns:
            Tensor: The transformed input tensor.
        """
        weight = weight.reshape(self.weight_shape)
        if self.channel_wise:
            return (x * weight)
        else:
            return (x.unsqueeze(-2) @ weight).squeeze(-2)

# class WithScalarsLinear(nn.Module):
# # class InvEquivLinear(nn.Module):
#     '''
#     Invariant-Equivariant Linear
#     '''
#     def __init__(self,
#                  linear: Union[SO3Linear, SO2Linear],
#                  scalars_in: int = 0,
#                  scalars_out: int = 0,
#                  external_scalar_weight: bool = False,
#     ):
#         super().__init__()
#         if isinstance(linear, SO3Linear):
#             self.linear_type = 'SO3'
#         elif isinstance(linear, SO2Linear):
#             self.linear_type = 'SO2'
#         else:
#             assert '`linear` should be one of SO3Linear or SO2Linear modules.'
#         assert not linear.channel_wise, 'Do not support channel wise linear operations.'
#         assert scalars_in + scalars_out != 0, 'At least one of `scalars_in` and `scalars_out` should be non-zero'
        
#         self.linear_ee = linear
#         self.scalars_in = scalars_in
#         self.scalars_out = scalars_out
#         self.external_scalar_weight = external_scalar_weight

#         if self.scalars_out * self.scalars_in != 0:
#             if external_scalar_weight:
#                 self.linear_ii = ElementWiseLinear(scalars_in, scalars_out)
#             else:
#                 self.linear_ii = nn.Linear(scalars_in, scalars_out)
#         else:
#             self.linear_ii = None
#         if self.linear_type == 'SO3':
#             if self.scalars_in != 0:
#                 self.linear_ie = SO3Linear(0, linear.L_edge, linear.L_out,
#                                         scalars_in, linear.out_channels,
#                                         external_scalar_weight, False)
#             else:
#                 self.linear_ie = None
#             if self.scalars_out != 0:
#                 self.linear_ei = SO3Linear(linear.L_in, linear.L_edge, 0,
#                                         linear.in_channels, scalars_out,
#                                         external_scalar_weight, False)
#             else:
#                 self.linear_ei = None
#         if self.linear_type == 'SO2':
#             if self.scalars_in != 0:
#                 self.linear_ie = SO2Linear(0, linear.L_out,
#                                         scalars_in, linear.out_channels,
#                                         external_scalar_weight, False)
#             else:
#                 self.linear_ie = None
#             if self.scalars_out != 0:
#                 self.linear_ei = SO2Linear(linear.L_in, 0,
#                                         linear.in_channels, scalars_out,
#                                         external_scalar_weight, False)
#             else:
#                 self.linear_ei = None

        
#         self.num_weights_ee = linear.num_weights if linear.external_weight else 0
#         self.dim_weights_ee = self.num_weights_ee * linear.in_channels * linear.out_channels

#         if self.external_scalar_weight:

#             if self.linear_ei is not None:
#                 self.num_weights_ei = self.linear_ei.num_weights
#             else:
#                 self.num_weights_ei = 0
#             self.dim_weights_ei = self.num_weights_ei * self.linear_ei.in_channels * self.linear_ei.out_channels

#             if self.linear_ie is not None:
#                 self.num_weights_ie = self.linear_ie.num_weights
#             else:
#                 self.num_weights_ie = 0
#             self.dim_weights_ie = self.num_weights_ie * self.linear_ie.in_channels * self.linear_ie.out_channels

#             if self.linear_ii is not None:
#                 self.num_weights_ii = 1
#             else:
#                 self.num_weights_ii = 0
#             self.dim_weights_ii = self.num_weights_ii * self.linear_ii.in_channels * self.linear_ii.out_channels


#     def split_weight(self, weight: Tensor):
#         return weight.split([
#             self.dim_weights_ee,
#             self.dim_weights_ei,
#             self.dim_weights_ie,
#             self.dim_weights_ii
#         ], dim=-3)
    
#     def forward(self, x_e:Tensor, x_i: Tensor = None, weight: Tensor = None):
#         w_ee, w_ei, w_ie, w_ii = 