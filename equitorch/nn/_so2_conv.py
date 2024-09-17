import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from ..typing import DegreeRange

# class SO2Conv(MessagePassing):

#     def __init__(self, 
#                  in_channels: int,
#                  out_channels: int,
#                  L_in: DegreeRange,
#                  L_out: DegreeRange,
#                  **kwargs
#                  ):
#         super().__init__(node_dim=-3, **kwargs)
#         self.
        