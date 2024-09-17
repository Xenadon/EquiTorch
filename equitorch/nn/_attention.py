from math import sqrt
from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import scatter, segment

from ._linear import SO2Linear

from ..typing import DegreeRange

class AttentionAggregation(nn.Module): # only computes the attention weights

    def __init__(
            self,
            L: DegreeRange,
            L_v: DegreeRange,
            embed_dim: int,
            kdim: int=None, 
            vdim: int=None,
            weight_v_producer: nn.Module = None,
    ):
        super().__init__()
        self.L = L
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.L_v = L_v if L_v is not None else L
        self.L_k = L_k if L_k is not None else L
        self.weight_v_producer = weight_v_producer

        self.linear_k = SO2Linear(L, L_k, embed_dim, kdim, True,)

    def forward():

    

# class MultiheadAttention(nn.Module):

#     def __init__(
#             self,
#             L: DegreeRange,
#             embed_dim: int,
#             num_heads: int, 
#             kdim: int=None, 
#             vdim: int=None,
#     ):
#         super().__init__()
#         self.L = L
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.kdim = kdim
#         self.vdim = vdim
    