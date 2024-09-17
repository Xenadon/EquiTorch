from math import sqrt
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import segment

from ..utils._indices import extract_batch_ptr, check_degree_range

from ..utils._clebsch_gordan import coo_CG

from ..math import dot
from ..typing import DegreeRange

r'''
fully_connected: C1, C2 -> C3, W[C1,C2,C3]

    Z_c3 = X_c1 \otimes_{W_{c1,c2,c3}} Y_c2

channel_wise_connected: C1=C2 -> C3, W[C,C3]

    Z_c3 = X_c \otimes_{W_{c,c3}} Y_c

pair_wise: C1,C2 -> C1C2, W[C1, C2] # for no-weight-tp

    Z_{c1,c2} = X_c1 \otimes_{W_{c1,c2}} Y_c2 

channel_wise: C1=C2=C3, W[C] # for no-weight-tp

    Z_{c} = X_c \otimes_{W_{c}} Y_c 

'''

def _weighted_tensor_product_fc(
    x1: Tensor, 
    x2: Tensor,
    weight: Tensor,
    CG_vals: Tensor,
    Ml1l2_ptr: Tensor, 
    l_ind: Tensor, 
    M1: Tensor,
    M2: Tensor,
    M_ptr: Tensor
):
    r'''
    fully_connected: C1, C2 -> C3, W[C1,C2,C3]

    Z_c3 = X_c1 \otimes_{W_{c1,c2,c3}} Y_c2
    '''
    x1 = x1.index_select(-2, M1).unsqueeze(-1) # B * MM1M2 * C1 * 1
    x2 = x2.index_select(-2, M2).unsqueeze(-2) # B * MM1M2 * 1 * C2
    # C_val: MM1M2
    inter = x2 * x1 * CG_vals.unsqueeze(-1).unsqueeze(-2) # B * MM1M2 * C1 * C2
    inter = segment(inter, Ml1l2_ptr.unsqueeze(0))  # B * Ml1l2 * C1 * C2
    weight = weight.index_select(dim=1,index=l_ind) # B * Ml1l2 * C1 * C2 * C3 
    ret = (inter.flatten(-2,-1).unsqueeze(-2) @ weight.flatten(-3,-2) ).squeeze(-2)   
        # B * Ml1l2 * 1 * C1C2 @ B * Ml1l2 * C1C2 * C3 -> B * Ml1l2 (* 1) * C3
    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C
    return ret

def _weighted_tensor_product_cwc(
    x1: Tensor, 
    x2: Tensor,
    weight: Tensor,
    CG_vals: Tensor,
    Ml1l2_ptr: Tensor, 
    l_ind: Tensor, 
    M1: Tensor,
    M2: Tensor,
    M_ptr: Tensor
):
    r'''
    channel_wise_connected: C1=C2 -> C3, W[C,C3]

    Z_c3 = X_c \otimes_{W_{c,c3}} Y_c
    '''
    x1 = x1.index_select(-2, M1) # B * MM1M2 * C
    x2 = x2.index_select(-2, M2) # B * MM1M2 * C
    # C_val: MM1M2
    inter = x2 * x1 * CG_vals.unsqueeze(-1) # B * MM1M2 * C
    inter = segment(inter, Ml1l2_ptr.unsqueeze(0))  # B * Ml1l2 * C
    weight = weight.index_select(dim=1,index=l_ind) # B * Ml1l2 * C * C3 
    ret = (inter.unsqueeze(-2) @ weight).squeeze(-2)   
        # B * Ml1l2 * 1 * C @ B * Ml1l2 * C * C3 -> B * Ml1l2 (* 1) * C3
    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C
    return ret

def _weighted_tensor_product_pw(
    x1: Tensor, 
    x2: Tensor,
    weight: Tensor,
    CG_vals: Tensor,
    Ml1l2_ptr: Tensor, 
    l_ind: Tensor, 
    M1: Tensor,
    M2: Tensor,
    M_ptr: Tensor
):
    r'''
    pair_wise: C1,C2 -> C1C2, W[C1, C2] 

    Z_{c1,c2} = X_c1 \otimes_{W_{c1,c2}} Y_c2 
    '''
    x1 = x1.index_select(-2, M1).unsqueeze(-1) # B * MM1M2 * C1 * 1
    x2 = x2.index_select(-2, M2).unsqueeze(-2) # B * MM1M2 * 1 * C2
    # C_val: MM1M2
    inter = x2 * x1 * CG_vals.unsqueeze(-1).unsqueeze(-2) # B * MM1M2 * C1 * C2
    inter = segment(inter, Ml1l2_ptr.unsqueeze(0))  # B * Ml1l2 * C1 * C2
    weight = weight.index_select(dim=1,index=l_ind) # B * Ml1l2 * C1 * C2 
    ret = inter * weight # B * Ml1l2 * C1 * C2 
    ret = segment(ret, M_ptr.unsqueeze(0)) # B * M * C1 * C2
    return ret

def _weighted_tensor_product_cw(x1: Tensor, 
              x2: Tensor,
              weight: Tensor,
              CG_vals: Tensor,
              l_ind_M1M2: Tensor, 
              M1: Tensor,
              M2: Tensor,
              M_ptr_M1M2: Tensor
              ):
    '''
        x1: B * M1 * C
        x2: B * M2 * C
        weight: B * ll1l2 * C
    '''
    x1 = x1.index_select(-2, M1) # B * MM1M2 * C
    x2 = x2.index_select(-2, M2) # B * MM1M2 * C
    # C_val: MM1M2
    inter = x2 * x1 * CG_vals.unsqueeze(-1) # B * MM1M2 * C
    weight = weight.index_select(dim=1,index=l_ind_M1M2) # B * MM1M2 * C
    ret = inter * weight # B * MM1M2 * C
    ret = segment(ret, M_ptr_M1M2.unsqueeze(0)) # B * M * C
    return ret

class WeightedTensorProduct(nn.Module):
    '''
    Weighted Tensor Product
    '''
    def __init__(self, 
                 L_in1: DegreeRange, 
                 L_in2:DegreeRange, 
                 L_out:DegreeRange,
                 in1_channels: int,
                 in2_channels: int,
                 out_channels: int = None,
                 connected: bool = False,
                 channel_wise: bool = True,
                 tp_type: str = None,
                 external_weight: bool = False, 
                ):
        super().__init__()
        if tp_type == 'cw' or tp_type == 'channel_wise':
            connected = False
            channel_wise = True
        elif tp_type == 'pw' or tp_type == 'pair_wise':
            connected = False
            channel_wise = False
        elif tp_type == 'cwc' or tp_type == 'channel_wise_connected':
            connected = True
            channel_wise = True
        elif tp_type == 'fc' or tp_type == 'fully_connected':
            connected = True
            channel_wise = False
        elif tp_type is not None:
            assert f"tp_type should be in [channel_wise, cw, pair_wise, pw, channel_wise_connected, cwc, fully_connected, fc] or None (not specified), got {tp_type}"
        
        if channel_wise:
            assert in1_channels == in2_channels, \
                f"in_channels should be same when channel_wise, but got {(in1_channels, in2_channels)}"
        if not connected and out_channels is not None:
            assert "Do not specify out_channels when connected is False"
        if connected and out_channels is None:
            assert "Please specify out_channels when connected is True"
        self.L_in1 = check_degree_range(L_in1)
        self.L_in2 = check_degree_range(L_in2)
        self.L_out = check_degree_range(L_out)
        self.external_weight = external_weight
        self.channel_wise = channel_wise
        self.connected = connected
        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.out_channels = out_channels if out_channels is not None \
            else in1_channels if channel_wise \
            else (in1_channels,in2_channels)
        
        if self.channel_wise and self.connected:
            self.tp_type = 'channel_wise_connected'
        elif self.channel_wise and not self.connected:
            self.tp_type = 'channel_wise'
        elif not self.channel_wise and self.connected:
            self.tp_type = 'fully_connected'
        else:
            self.tp_type = 'pair_wise'
        
        CG_vals, Ms, ls_cg, l_ind_cg = coo_CG(L_out, L_in1, L_in2)

        if self.channel_wise and not self.connected: # channel_wise
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
                    torch.randn(1, self.num_weights,self.in_channels))
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
                if self.channel_wise: # cwc
                    self.weight = nn.Parameter(
                        torch.randn(1, self.num_weights,
                                    self.in1_channels, self.out_channels) \
                                    * sqrt(2/(self.in1_channels + self.out_channels)))
                elif self.connected: # fc
                    self.weight = nn.Parameter(
                        torch.randn(
                            1, self.num_weights,
                            self.in1_channels, self.in2_channels, self.out_channels) \
                                    * sqrt(2/((self.in1_channels * self.in2_channels) + self.out_channels)))
                else: # pw
                    self.weight = nn.Parameter(
                        torch.randn(
                            1, self.num_weights,
                            self.in1_channels, self.in2_channels) \
                            * sqrt(2/(self.in1_channels * self.in2_channels)))
                    
    def forward(self, x1: Tensor, x2: Tensor, weight: Optional[Tensor] = None, **kwargs): # !TO REMOVE
        if weight is None and not self.external_weight:
            weight = self.weight
        if self.channel_wise and self.connected:
            return _weighted_tensor_product_cwc(x1, x2, weight,self.CG_vals, self.Ml1l2_ptr, self.l_ind, self.M1, self.M2, self.M_ptr)
        elif self.channel_wise and not self.connected:
            return _weighted_tensor_product_cw(x1, x2, weight,self.CG_vals, self.l_ind, self.M1, self.M2, self.M_ptr)
        elif not self.channel_wise and self.connected:
            return _weighted_tensor_product_fc(x1, x2, weight,self.CG_vals, self.Ml1l2_ptr, self.l_ind, self.M1, self.M2, self.M_ptr)
        else:
            return _weighted_tensor_product_pw(x1, x2, weight,self.CG_vals, self.Ml1l2_ptr, self.l_ind, self.M1, self.M2, self.M_ptr)

    def __repr__(self) -> str:
        type_repr = ''
        type_repr += f'tp_type={self.tp_type}, '
        if self.channel_wise and self.connected:
            type_repr += f'in_channels={self.in1_channels}, out_channels={self.out_channels}'
        elif self.channel_wise and not self.connected:
            type_repr += f'in_channels={self.in1_channels}'
        elif not self.channel_wise and self.connected:
            type_repr += f'in1_channels={self.in1_channels}, in2_channels={self.in2_channels}, out_channels={self.out_channels}'
        else:
            type_repr += f'in1_channels={self.in1_channels}, in2_channels={self.in2_channels}'
        return f'{self.__class__.__name__}(\n  L_in1={self.L_in1}, L_in2={self.L_in2}, L_out={self.L_out}, \n  {type_repr}, external_weight={self.external_weight}\n)'

def _tensor_product_cw(x1: Tensor, 
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
    inter = x2 * x1 * CG_vals.unsqueeze(-1) # B * MM1M2 * C
    ret = segment(inter, M_ptr.unsqueeze(0)) # B * M * C
    return ret

def _tensor_product_pw(x1: Tensor, 
              x2: Tensor,
              CG_vals: Tensor,
              M1: Tensor,
              M2: Tensor,
              M_ptr: Tensor, ):
    '''
        x1: B * M1 * C1
        x2: B * M2 * C2
    '''
    x1 = x1.index_select(-2, M1).unsqueeze(-1) # B * MM1M2 * C1 * 1
    x2 = x2.index_select(-2, M2).unsqueeze(-2) # B * MM1M2 * 1 * C2 
    inter = x2 * x1 * CG_vals.unsqueeze(-1).unsqueeze(-2) # B * MM1M2 * C1 * C2
    ret = segment(inter, M_ptr.unsqueeze(0)) # B * M * C1 * C2
    return ret

class TensorProduct(nn.Module):
    
    def __init__(self, 
                 L_in1: DegreeRange, 
                 L_in2:DegreeRange, 
                 L_out:DegreeRange, 
                 channel_wise: bool =True):
        super().__init__()
        self.L_in1 = check_degree_range(L_in1)
        self.L_in2 = check_degree_range(L_in2)
        self.L_out = check_degree_range(L_out)
        self.channel_wise = channel_wise
        CG_vals, Ms, ls_cg, l_ind_cg = coo_CG(L_out, L_in1, L_in2)
        M_batch, M_ptr, M = extract_batch_ptr([Ms[0]])

        self.register_buffer('M1', Ms[1])
        self.register_buffer('M2', Ms[2])
        self.register_buffer('M_ptr', M_ptr)
        self.register_buffer('CG_vals', CG_vals)

    def forward(self, x1: Tensor, x2: Tensor, **kwargs): # !TO REMOVE
        if self.channel_wise:
            return _tensor_product_cw(x1, x2, self.CG_vals, self.M1, self.M2, self.M_ptr)
        else:
            return _tensor_product_pw(x1, x2, self.CG_vals, self.M1, self.M2, self.M_ptr)
        
    def __repr__(self):
        return f'{self.__class__.__name__}(L_in1={self.L_in1}, L_in2={self.L_in2}, L_out={self.L_out}, channel_wise={self.channel_wise})'

class TensorDot(nn.Module):
    r"""
    The module that computes the degree-wise dot product between spherical features.

    For spherical features :math:`x_1 = [{x_1}_m^l]_{c1}` and :math:`x_2 = [{x_2}_m^l]_{c2}`,
    this module computes:

    .. math::

        d_c^l = \sum_{m=-l}^l {{x_1}_m^l}_c {{x_2}_m^l}_c

    if channel_wise, or

    .. math::

        d_{c_1,c_2}^l = \sum_{m=-l}^l {{x_1}_m^l}_{c1} {{x_2}_m^l}_{c2}
        
    if not channel_wise.

    Parameters
    ----------
    L : DegreeRange
        Range of degrees to consider in the dot product computation.
    channel_wise : bool, optional
        If True, compute channel-wise dot product. Default is True.

    Attributes
    ----------
    L : DegreeRange
        The range of degrees used in the dot product computation.
    channel_wise : bool
        Flag indicating whether to compute channel-wise dot product.

    Examples
    --------
    >>> L = DegreeRange(0, 3)
    >>> tensor_dot = TensorDot(L, channel_wise=True)
    >>> x1 = torch.randn(32, 16, 64)  # (N, num_orders, C1)
    >>> x2 = torch.randn(32, 16, 64)  # (N, num_orders, C2)
    >>> result = tensor_dot(x1, x2)
    >>> print(result.shape)
    torch.Size([32, 4, 64])  # (N, num_degrees, C)

    See Also
    --------
    dot : The underlying function used for dot product computation.

    Notes
    -----
    - The forward pass computes the dot product using the specified degree range and channel-wise setting.
    """
    def __init__(self, 
                 L: DegreeRange,
                 channel_wise: bool = True
                ):
        super().__init__()
        self.L = L
        self.channel_wise = channel_wise

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        r"""
        Compute the degree-wise dot product between two input tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            First input tensor of shape (N, num_orders, C1).
        x2 : torch.Tensor
            Second input tensor of shape (N, num_orders, C2).

        Returns
        -------
        torch.Tensor
            The result of the dot product computation. 
            If channel_wise is True:
                Shape is (N, num_degrees, C)
            If channel_wise is False:
                Shape is (N, num_degrees, C1, C2)

        Notes
        -----
        - N is the batch size
        - num_orders is the total number of spherical harmonic orders
        - num_degrees is the number of degrees in the specified range L
        - C, C1, C2 are the number of channels
        """
        return dot(x1, x2, self.L, self.channel_wise)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(L={self.L}, channel_wise={self.channel_wise})'
