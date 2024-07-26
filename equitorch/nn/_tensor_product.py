from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import segment

from ..utils._indices import extract_batch_ptr

from ..utils._clebsch_gordan import coo_CG

from ..math import clebsch_gordan, dot
from ..typing import DegreeRange
from ..utils import (
    check_degree_range
)


def _weighted_tensor_product(x1: Tensor, 
              x2: Tensor,
              weight: Tensor,
              CG_vals: Tensor,
              M_ptr: Tensor, 
              l_ind: Tensor, 
              M1: Tensor,
              M2: Tensor,
              ):
    '''
        x1: B * M1 * C
        x2: B * M2 * C
        weight: B * ll1l2 * C
    '''
    x1 = x1.index_select(-2, M1) # B * MM1M2 * C
    x2 = x2.index_select(-2, M2) # B * MM1M2 * C
    # C_val: MM1M2
    inter = x2 * x1 * CG_vals.unsqueeze(0) # B * MM1M2 * C
    weight = weight.index_select(dim=1,index=l_ind) # B * MM1M2 * C
    ret = inter * weight # B * MM1M2 * C'
    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C
    return ret

class WeightedTensorProduct(nn.Module):

    def __init__(self, 
                 L_in1: DegreeRange, 
                 L_in2:DegreeRange, 
                 L_out:DegreeRange,
                 external_weight: bool = False, 
                 **kwargs):
        self.L_in1 = check_degree_range(L_in1)
        self.L_in2 = check_degree_range(L_in2)
        self.L_out = check_degree_range(L_out)
        self.external_weight = external_weight

        CG_vals, Ms, ls_cg, l_ind = coo_CG(L_out, L_in1, L_in2)

        M_batch, M_ptr, M = extract_batch_ptr([Ms[0]])

        self.register_buffer('Ms', Ms)
        self.register_buffer('M1', Ms[1])
        self.register_buffer('M2', Ms[2])
        self.register_buffer('l_ind', l_ind)
        self.register_buffer('M_ptr', M_ptr)
        self.register_buffer('CG_vals', CG_vals)

        self.num_weights = l_ind.unique().numel()

        if not self.external_weight:
            self.weight = nn.Parameter(
                torch.randn(1, self.num_weights) 
            )

        
    def forward(self, x1: Tensor, x2: Tensor, weight: Optional[Tensor] = None,):
        if weight is None and not self.external_weight:
            weight = self.weight
        return _weighted_tensor_product(x1, x2, weight,self.CG_vals, self.M_ptr, self.l_ind, self.M1, self.M2)
    

def _tensor_product(x1: Tensor, 
              x2: Tensor,
              CG_vals: Tensor,
              M1: Tensor,
              M2: Tensor,
              M_ptr: Tensor, ):
    '''
        x1: B * M1 * C
        x2: B * M2 * C
    '''
    x1 = x1.index_select(-2, M1) # B * MM1M2 * C
    x2 = x2.index_select(-2, M2) # B * MM1M2 * C
    inter = x2 * x1 * CG_vals.unsqueeze(0) # B * MM1M2 * C
    ret = segment(inter, M_ptr.unsqueeze(0)) # B * M * C
    return ret

class TensorProduct(nn.Module):
    
    def __init__(self, L_in1: DegreeRange, L_in2:DegreeRange, L_out:DegreeRange):
        self.L_in1 = check_degree_range(L_in1)
        self.L_in2 = check_degree_range(L_in2)
        self.L_out = check_degree_range(L_out)
        CG_vals, Ms, ls_cg, l_ind_cg = coo_CG(L_out, L_in1, L_in2)
        M_batch, M_ptr, M = extract_batch_ptr([Ms[0]])

        self.register_buffer('M1', Ms[1])
        self.register_buffer('M2', Ms[2])
        self.register_buffer('M_ptr', M_ptr)
        self.register_buffer('CG_vals', CG_vals)

    def forward(self, x1: Tensor, x2: Tensor):
        if weight is None and not self.external_weight:
            weight = self.weight
        return _tensor_product(x1, x2, self.CG_vals, self.M1, self.M2, self.M_ptr)
    
   
class ScalarTensorProduct(nn.Module):

    def __init__(self, 
                 channels: int, 
                 L_in: DegreeRange, 
                 weighted: bool = True,
                 external_weight: bool = False
                 ):

        self.L_in = check_degree_range(L_in)
        self.channels = channels
        self.weighted = weighted
        self.external_weight = external_weight

        if weighted:

            self.num_weights = self.L_in[1] - self.L_in[0]+1
            
            if not external_weight:
                self.weight = nn.Parameter(
                    torch.randn(1, self.num_weights,
                                self.in_channels))
        else:
            self.num_weights = 0



    def forward(self, x1: Tensor, x2: Tensor):
        if self.weighted and weight is None and not self.external_weight:
            weight = self.weight
        d = dot(x1, x2, self.L_in)
        if self.weighted:
            return d * self.weight
        else:
            return d