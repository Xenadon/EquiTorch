from typing import Callable, Optional, Union, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.typing import Adj

from ..utils._indices import check_degree_range
from ..utils._geometries import rot_on

from ._activation import NormAct, ShiftedSoftPlus

from ..typing import DegreeRange

from ._linear import DegreeWiseLinear, SO3Linear, SO2Linear

class TFNBlock(MessagePassing):
    '''
    The block of Tensor Field Networks in *Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds* 
    [https://arxiv.org/abs/1802.08219]
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 L_in: DegreeRange, 
                 L_edge: DegreeRange, 
                 L_out: DegreeRange,
                 channel_wise: bool = True,
                 weight_producer: Callable = None,
                 act: Callable = None,
                ):
        super().__init__(node_dim=0)
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)
        self.L_edge = check_degree_range(L_edge)

        self.lin = SO3Linear(L_in, L_edge, L_out, 
                             in_channels, out_channels, 
                             external_weight=True, channel_wise=channel_wise)
        self.lin_weight_shape = (-1, self.lin.num_weights, in_channels) if channel_wise \
            else (-1, self.lin.num_weights, in_channels, out_channels)
        # self.self_int = DegreeWiseLinear(self.L_in, self.L_out, in_channels, out_channels)
        self.self_int = DegreeWiseLinear(self.L_out, self.L_out, out_channels, out_channels)
        self.act = act
        self.weight_producer = weight_producer

    def forward(self, x: Tensor, edge_index: Adj,
                edge_feat: Tensor, edge_emb: Tensor, edge_weight: Tensor):
        lin_weight = edge_emb if self.weight_producer is None else self.weight_producer(edge_emb)
        lin_weight = lin_weight.view(*(self.lin_weight_shape))
        out = self.propagate(edge_index, x=x, edge_feat=edge_feat,
                             lin_weight=lin_weight, edge_weight=edge_weight)
        out = self.self_int(out)
        return self.act(out) if self.act is not None else out
    
    def message(self, x_j:Tensor, edge_feat:Tensor, lin_weight:Tensor, edge_weight:Tensor):
        x_j = self.lin(x_j, edge_feat, lin_weight)
        return edge_weight.view(-1,1,1) * x_j
    

class SO2TFNBlock(MessagePassing):
    '''
    The block of Tensor Field Networks in *Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds*,
    [https://arxiv.org/abs/1802.08219]
    but replace the SO(3) convolutions to the SO(2) ones in *Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs*
    [https://arxiv.org/abs/2302.03655]
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 L_in: DegreeRange, 
                 L_out: DegreeRange,
                 channel_wise: bool = True,
                 weight_producer: Callable = None,
                 act: Callable = None,
                ):
        super().__init__(node_dim=0)
        self.L_in = check_degree_range(L_in)
        self.L_out = check_degree_range(L_out)

        self.lin = SO2Linear(L_in, L_out, 
                             in_channels, out_channels, 
                             external_weight=True, channel_wise=channel_wise)
        self.lin_weight_shape = (-1, self.lin.num_weights, in_channels) if channel_wise \
            else (-1, self.lin.num_weights, in_channels, out_channels)
        self.self_int = DegreeWiseLinear(self.L_out, self.L_out, out_channels, out_channels)
        self.act = act
        self.weight_producer = weight_producer

    def forward(self, x: Tensor, edge_index: Adj,
                D_in: Tensor, DT_out: Tensor, edge_emb: Tensor, edge_weight: Tensor):
        lin_weight = edge_emb if self.weight_producer is None else self.weight_producer(edge_emb)
        lin_weight = lin_weight.view(*(self.lin_weight_shape))
        out = self.propagate(edge_index, x=x, D_in=D_in, DT_out=DT_out,
                             lin_weight=lin_weight, edge_weight=edge_weight)
        out = self.self_int(out)
        return self.act(out) if self.act is not None else out
    
    def message(self, x_j:Tensor, 
                lin_weight:Tensor, edge_weight:Tensor,
                D_in:Tensor, DT_out:Tensor):
        x_j = rot_on(D_in, x_j)
        x_j = self.lin(x_j, lin_weight)
        x_j = rot_on(DT_out, x_j)
        return edge_weight.view(-1,1,1) * x_j