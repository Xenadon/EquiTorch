from math import sqrt
from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing

from ._linear import SO2Linear
from ..utils._indices import degrees_in_range
from ..utils._geometries import rot_on


from ..typing import DegreeRange

# def _transpose_multihead(x: Tensor, num_heads: int):
#     '''
#     x: N * M * C
#     reshaped: (num_heads * N) * M * C//num_heads
#     '''
#     return x.unflatten(-1, (num_heads, -1)).permute(-2,*x.shape[0:-2], -1).flatten(0,1)

# def reshape_weight(x):
#     '''
#     weight: N * num_weights * C_in * C_out
#     reshaped: (num_heads * N) * num_weights * C_in//num_heads * C_out
#     '''
class ScaledDotAttention(nn.Module):  # only computes the attention weights
    '''
    Input: 
        x_s & x_t: num_edges, num_orders, num_channels
        edge_index
    '''
    def __init__(
            self,
            L_in: DegreeRange,
            L_k: DegreeRange,
            in_channels: int,
            k_channels: int,
            num_heads: int = 1,
            weight_k_producer: nn.Module = None,
            weight_q_producer: nn.Module = None
    ):
        super().__init__()
        self.L_in = L_in
        self.L_k = L_k
        self.in_channels = in_channels
        self.k_channels = k_channels
        self.num_heads = num_heads
        self.proj_k = SO2Linear(L_in, L_k, in_channels, num_heads*k_channels, weight_k_producer is None, False)
        self.proj_q = SO2Linear(L_in, L_k, in_channels, num_heads*k_channels, weight_q_producer is None, False)
        self.weight_k_producer = weight_k_producer
        self.weight_k_shape = (-1, self.proj_k.num_weights, in_channels, num_heads*k_channels) if weight_k_producer is not None else ()
        self.weight_q_producer = weight_q_producer
        self.weight_q_shape = (-1, self.proj_q.num_weights, in_channels, num_heads*k_channels) if weight_q_producer is not None else ()

    def forward(self, x_q:Tensor, x_k:Tensor, edge_emb: Tensor=None, index: Tensor = None, ptr: Tensor = None, num_nodes: int = None):
        q = self.proj_q.forward(x_q, self.weight_q_producer(edge_emb).reshape(self.weight_q_shape)\
                                 if self.weight_q_producer is not None else None).unflatten(-1, self.num_heads, self.k_channels)
        k = self.proj_k.forward(x_k, self.weight_k_producer(edge_emb).reshape(self.weight_k_shape)\
                                 if self.weight_k_producer is not None else None).unflatten(-1, self.num_heads, self.k_channels) 
        pre_softmax = (q*k).sum((-1,-3)) * (self.k_channels ** -0.5)
        return softmax(pre_softmax, index = index, ptr = ptr, num_nodes=num_nodes, dim=0)

class AttentionalBlock(MessagePassing):


    def __init__(
            self,
            L_in: DegreeRange,
            L_out: DegreeRange,
            in_channels: int,
            k_channels: int,
            out_channels: int,
            num_heads: int = 1,
            attention_score_producer: nn.Module = None,
            v_producer: nn.Module = None,
            weight_q_producer: nn.Module = None,
            weight_k_producer: nn.Module = None,
            weight_v_producer: nn.Module = None,
    ):
        super().__init__()
        self.L_in = L_in
        self.L_out = L_out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dim_v = out_channels // num_heads
        self.attention_score_producer = attention_score_producer if attention_score_producer is not None\
            else ScaledDotAttention(L_in, L_in, in_channels, k_channels, num_heads, weight_k_producer, weight_q_producer)
        self.v_producer = v_producer if v_producer is not None\
            else SO2Linear(L_in, L_out, in_channels, out_channels, weight_v_producer is not None, False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_emb = None, D_in=None, DT_out=None):
        if D_in is not None:
            x = rot_on(D_in, x)
        x1 = self.propagate(edge_index=edge_index, x=x, edge_emb=edge_emb)
        if DT_out is not None:
            x1 = rot_on(DT_out, x1)
        return x1
    def message(self, x_j, x_i, edge_emb, edge_index):
        # x_j to x_i, i: query, j: key
        att = self.attention_score_producer.forward(x_i, x_j, edge_emb, edge_index[1]).unsqueeze(-1).unsqueeze(1)
        # att: E * 1 * num_heads * 1
        weight_v = self.v_producer(edge_emb) if self.v_producer is not None else edge_emb
        v = self.v_producer.forward(x_j, weight_v).unflatten(-1, (self.num_heads, self.dim_v))
        # E * M * num_heads * dim_v
        return (att * v).flatten(-2, -1)

# class SE3TransformerBlock(nn.Module): # only computes the attention weights

#     def __init__(
#             self,
#             L: DegreeRange,
#             L_k: DegreeRange,
#             L_v: DegreeRange,
#             embed_dim: int,
#             num_heads: int = 1,
#             kdim: int=None, 
#             vdim: int=None,
#             weight_v_producer: nn.Module = None,
#             weight_k_producer: nn.Module = None,
#     ):
#         super().__init__()
#         self.L = L
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self.num_heads = num_heads
#         self.L_v = L_v if L_v is not None else L
#         self.L_k = L_k if L_k is not None else L
#         self.weight_v_producer = weight_v_producer
#         self.weight_k_producer = weight_k_producer
#         self.shape_v = (-1, num_orders_in(L_v), num_heads, vdim)
#         self.shape_k = (-1, num_orders_in(L_k), num_heads, kdim)

#         self.linear_k = SO2Linear(L, L_k, embed_dim, kdim, True)
#         self.linear_k = SO2Linear(L, L_k, embed_dim, kdim, True,)

#     def forward():

    

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
    