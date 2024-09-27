from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing 

from ._radial_basis import GaussianBasisExpansion
from ._activation import shifted_softplus

class CFConv(MessagePassing):

    def __init__(self, edge_channels: int, 
                 hidden_channels: int, 
                 out_channels: int,
                ):
        super().__init__(node_dim=-3, aggr='sum')
        self.lin1 = nn.Linear(edge_channels, hidden_channels, bias=False)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x:Tensor, edge_index: Tensor, edge_attr: Tensor, edge_weight: Tensor = None) -> Tensor:
        x = self.propagate(x = x, edge_index=edge_index, edge_attr=edge_attr, edge_weight = edge_weight)
        return x
    
    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        W = self.lin1(edge_attr)
        W = shifted_softplus(W)
        W = self.lin2(W)
        W = shifted_softplus(W)
        W.reshape(-1)
        if edge_weight is None:
            return x_j * W
        else:
            return x_j * W * edge_weight

class SchNetBlock(nn.Module):

    def __init__(self, 
                 hidden_channels: int, 
                 edge_channels: int, 
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None):
        if in_channels is None:
            in_channels = hidden_channels
        if out_channels is None:
            out_channels = out_channels
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.cfconv = CFConv(edge_channels, hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_weight: Optional[Tensor] = None):
        v = self.lin1(x)
        v = self.cfconv(v, edge_index, edge_attr, edge_weight)
        v = self.lin2(v)
        v = shifted_softplus(v)
        v = self.lin3(v)
        return x + v
    

    
    
        