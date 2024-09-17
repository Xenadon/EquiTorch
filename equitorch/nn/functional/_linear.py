from math import sqrt
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import scatter

from ...typing import DegreeRange
from ...utils import expand_degree_to_order

def dw_linear(x:Tensor, weight:Tensor, l_min:int, l_max:int) -> Tensor:
    """Perform degree wise linear transformation.

    :param x: input feature of shape :math:`(B,M,C_{in})`
    :type x: Tensor
    :param weight: weights of shape :math:`(B,L,C_{out},C_{in})` 
    :type weight: Tensor
    :param l_min: the minimal degree
    :type l_min: int
    :param l_max: the maximal degree
    :type l_max: int
    :return: transformed feature of shape :math:`(B,M,C_{out})`
    :rtype: Tensor
    """
    return (x.unsqueeze(-2) @ expand_degree_to_order(
                  weight.unsqueeze(0), l_min, l_max, dim=-2)
            ).squeeze(-2)

def so3_linear(x:Tensor, weight:Tensor, l_min:int, l_max:int)

def so2_linear(x: Tensor, weight:Tensor, weight_index: Tensor, from_indices: Tensor, to_indices: Tensor) -> Tensor:
    # W: Num * Ci * Co
    W = weight.index_select(dim=0, index=weight_index) 
    # X: N * Num * Ci 
    X = x.index_select(dim=-2,index=from_indices) 
    out = (X.unsqueeze(-2)@W.unsqueeze(0)).squeeze(-2)
    return scatter(out, index=to_indices, dim=-2)


# def so2_linear(x: Tensor, weight:Tensor, weight_index: Tensor, from_indices: Tensor, to_indices: Tensor) -> Tensor:
#     # W: Num * Ci * Co
#     W = weight.index_select(dim=0, index=weight_index) 
#     # X: N * Num * Ci 
#     X = x.index_select(dim=-2,index=from_indices) 
#     out = (X.unsqueeze(-2)@W.unsqueeze(0)).squeeze(-2)
#     return scatter(out, index=to_indices, dim=-2)
