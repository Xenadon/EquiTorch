from math import sqrt
from typing import Optional
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
    num_order_between
    # SO2_type_weight_prepare,
    # SO3_type_weight_prepare,
    # reduce_order_to_degree,
    # order_0_in
)
from ..utils._clebsch_gordan import coo_CG

class DWLinear(nn.Module):
    '''DegreeWiseLinear.
    '''

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 L: DegreeRange,
                 L_in: DegreeRange = None
                ):
        out_channels = in_channels if out_channels is None else out_channels 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L = check_degree_range(L)
        self.L_in = check_degree_range(L_in) if L_in is not None else L
        self.weight = self.self_interaction_weight = nn.Parameter(
                    torch.randn(self.L[1]+1 - self.L[0], in_channels, out_channels) * 2 / sqrt(in_channels + out_channels))

    def forward(self, x: Tensor):
        # x = x.view(-1, x.shape[-2], x.shape[-1])
        x = extract_in_degree(x, self.L_in, self.L)
        return (x.unsqueeze(-2) @ 
                expand_degree_to_order(self.weight.unsqueeze(0), self.l_min, self.l_max, dim=-2)).squeeze(-2)
    

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
            for l_out in range(L_out[0], L_out[1]+1)
            for l_in in range(L_in[0], L_in[1]+1)
            for m_weight in range(-min(l_out, l_in), min(l_out, l_in)+1)
        ) 
        for m_out, m_in in (((-m_weight,-abs(m_weight)), (m_weight,abs(m_weight))) if m_weight != 0 else ((0,0),))
    ])
    Ms = torch.tensor([[t[0], t[1]] for t in ret]).T
    ls = torch.tensor([[t[2], t[3]] for t in ret]).T
    weight_sign = torch.tensor([t[4] for t in ret])
    weight_idx = torch.tensor([t[5] for t in ret])
    return Ms, ls, weight_sign, weight_idx

class SO2Linear(nn.Module):

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 L_in: DegreeRange, 
                 L_out: DegreeRange, 
                 external_weight: bool = False,
                 channel_wise: bool = True
                 ):
        assert in_channels == out_channels or not channel_wise
        super().__init__()
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        self.in_channels = in_channels
        self.out_ms = num_order_between(*L_out)
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
        
        W = weight.index_select(dim=1, index=self.weight_index) * self.weight_sign.view(1,-1,1,1) # N * Num * Ci (* Co)
        X = x.index_select(dim=1,index=self.M_in) # N * Num * Ci 

        if self.channel_wise:
            out = X * W
        else:
            out = (X.unsqueeze(-2)@W).squeeze(-2)
        return scatter(out, index=self.M_out, dim=1, dim_size=self.out_ms)
    
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
            in_channels: int,
            out_channels: int,
            L_in: DegreeRange,
            L_edge: DegreeRange,
            L_out: DegreeRange,
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



class ScalarSO3Linear(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        L_in: DegreeRange,
        external_weight: bool = False,
        channel_wise: bool = True,
        **kwargs
    ):
        assert in_channels == out_channels or not channel_wise


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L_in = check_degree_range(L_in)
        self.external_weight = external_weight
        
        self.channel_wise = channel_wise

        self.num_weights = self.L_in[1] - self.L_in[0] + 1
        if not external_weight:
            if channel_wise:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.in_channels))
            else:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                in_channels, out_channels) \
                                    * sqrt(2/(in_channels+out_channels)))

    def forward(self, 
                feature: Tensor, 
                edge_feat: Tensor, 
                weight: Optional[Tensor] = None
        ):
        if weight is None and not self.external_weight:
            weight = self.weight

        d = dot(feature, edge_feat.unsqueeze(-1), self.L_in) # B * L * C
        
        
        if self.channel_wise:
            return d * weight
        else:
            return d @ weight
        