from math import sqrt
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import segment

from ..utils.indices import extract_batch_ptr, check_degree_range

from ..utils.clebsch_gordan import coo_CG

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
    r'''
    The weighted tensor product

    .. math::
        (\mathbf{x}\otimes_{\mathbf{W}}\mathbf{y}) = \bigoplus_{l\in L_{\text{out}}}\sum_{l_1,l_2}{\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}

    where the summation of :math:`(l_1,l_2)` is over all the values such that 
    :math:`l_1\in L_1, l_2\in L_2` and :math:`|l_1-l_2|\le l\le l_1+l_2`.

    The definition of :math:`{\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}`
    depends on :obj:`tp_type`:

    - Channel wise:

        When :obj:`tp_type=\'cw\'` or :obj:`connected = False and channel_wise = True`,
        :math:`\mathbf{W}_{l_1,l_2}^{l}` will be a :math:`(C,)` shaped tensor.
        :math:`{\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}`
        will have :math:`C` channels with each channel defined as

        .. math::
            {\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}_c
            =\mathbf{W}_{l_1,l_2;c}^{(l)}(\mathbf{x}^{(l_1)}_c\otimes \mathbf{y}^{(l_2)}_c)^{(l)}.

    - Pair wise:

        When :obj:`tp_type=\'pw\'` or :obj:`connected = False and channel_wise = False`,
        :math:`\mathbf{W}_{l_1,l_2}^{l}` will be a :math:`(C_1,C_2)` shaped tensor. 
        :math:`{\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}`
        will have :math:`C_1\cdot C_2` channels with each channel defined as

        .. math::
            {\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}_{c_1c_2}
            =\mathbf{W}_{l_1,l_2;c_1c_2}^{(l)}(\mathbf{x}^{(l_1)}_{c_1}\otimes \mathbf{y}^{(l_2)}_{c_2})^{(l)}.
            
    - Channel-wise connected:

        When :obj:`tp_type=\'cwc\'` or :obj:`connected = True and channel_wise = True`,
        :math:`\mathbf{W}_{l_1,l_2}^{l}` will be a :math:`(C,C_{\text{out}})` shaped tensor. 
        :math:`{\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}`
        will have :math:`C_{\text{out}}` channels with each channel defined as

        .. math::
            {\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}_{c_{\text{out}}}
            =\sum_{c=1}^{C}\mathbf{W}_{l_1,l_2;cc_{\text{out}}}^{(l)}(\mathbf{x}^{(l_1)}_{c}\otimes \mathbf{y}^{(l_2)}_{c})^{(l)}.
                   
    - Fully connected:

        When :obj:`tp_type=\'fc\'` or :obj:`connected = False and channel_wise = False`,
        :math:`\mathbf{W}_{l_1,l_2}^{l}` will be a :math:`(C_1,C_2,C_{\text{out}})` shaped tensor. 
        :math:`{\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}`
        will have :math:`C_{\text{out}}` channels with each channel defined as

        .. math::
            {\mathbf{W}_{l_1,l_2}^{l}}(\mathbf{x}^{(l_1)}\otimes \mathbf{y}^{(l_2)})^{(l)}_{c_{\text{out}}}
            =\sum_{c_1=1}^{C_1}\sum_{c_2=1}^{C_2}\mathbf{W}_{l_1,l_2;c_1c_2c_{\text{out}}}^{(l)}(\mathbf{x}^{(l_1)}_{c_1}\otimes \mathbf{y}^{(l_2)}_{c_2})^{(l)}.

    And in all the cases above, the tensor product of single-degree, single-channel inputs is defined as

    .. math::
        (\mathbf{x}^{(l_1)}_{c}\otimes \mathbf{y}^{(l_2)}_{c'})^{(l)}_m=\sum_{m_1=-l_1}^{l_1}\sum_{m_2=-l_2}^{l_2}C_{(l_1,m_1)(l_2,m_2)}^{(l,m)}
        \mathbf{x}^{(l_1)}_{m_1,c}\mathbf{y}^{(l_2)}_{m_2,c'}, m=-l,\dots,l,

    where :math:`C_{(l_1,m_1)(l_2,m_2)}^{(l,m)}` are the Clebsch-Gordan coefficients.

    Also, we allow different weights for different input pairs when :obj:`external_weights` is :obj:`True`.
    
    The equivariance of weighted tensor product means that for any rotation 
    :math:`\mathbf{R}\in\text{SO(3)}` and corresponding Wigner D matrices
    :math:`\mathbf{D}_1`, :math:`\mathbf{D}_2` and :math:`\mathbf{D}` on 
    the two input spaces and the output space, correspondingly, it holds

    .. math::
        [(\mathbf{D}_1\mathbf{x})\otimes_{\mathbf{W}}(\mathbf{D}_2\mathbf{y})]=\mathbf{D}(\mathbf{x}\otimes_{\mathbf{W}}\mathbf{y}).


    Note
    ----
    - When :obj:`channel_wise` is :obj:`True`, :obj:`in1_channels` must be equal to :obj:`in2_channels`.
    - When :obj:`connected` is :obj:`False`, :obj:`out_channels` must not be specified.
    - When :obj:`connected` is :obj:`True`, :obj:`out_channels` must be specified.

    Parameters
    ----------
    L_in1 : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the first input.
    L_in2 : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the second input.
    L_out : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the output.
    in1_channels : int
        The number of channels in the first input.
    in2_channels : int
        The number of channels in the second input. Must be the same as :obj:`in1_channels` if :obj:`channel_wise` is :obj:`True`.
    out_channels : int, optional
        The number of channels in the output. Must be specified if :obj:`connected` is :obj:`True`, and must not be specified if :obj:`connected` is :obj:`False`.
    connected : bool, optional
        Whether the weights are connected or not. Default is :obj:`False`. Will be overridden if :obj:`tp_type` is specified.
    channel_wise : bool, optional
        Whether the weights are channel-wise or not. Default is :obj:`True`. Will be overridden if :obj:`tp_type` is specified.
    tp_type : str, optional
        The type of tensor product. Can be one of the following:

            - :obj:`'channel_wise'` or :obj:`'cw'`: Channel-wise tensor product with :obj:`connected=False` and :obj:`channel_wise=True`.
            - :obj:`'pair_wise'` or :obj:`'pw'`: Pair-wise tensor product with :obj:`connected=False` and :obj:`channel_wise=False`.
            - :obj:`'channel_wise_connected'` or :obj:`'cwc'`: Channel-wise connected tensor product with :obj:`connected=True` and :obj:`channel_wise=True`.
            - :obj:`'fully_connected'` or :obj:`'fc'`: Fully connected tensor product with :obj:`connected=True` and :obj:`channel_wise=False`.
        
        If not provided, :obj:`connected` and :obj:`channel_wise` will be used to determine the type of tensor product.
    external_weights : bool, optional
        Whether to use external weights or not. Default is :obj:`False`.
    '''

    def __init__(self, 
                 L_in1: DegreeRange, 
                 L_in2:DegreeRange, 
                 L_out:DegreeRange,
                 in1_channels: int,
                 in2_channels: int,
                 out_channels: Optional[int] = None,
                 connected: bool = False,
                 channel_wise: bool = True,
                 tp_type: Optional[str] = None,
                 external_weights: bool = False, 
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
        self.external_weights = external_weights
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
            self.weight_shape = (self.num_weights, self.in1_channels)
            if not self.external_weights:
                self.weight = nn.Parameter(torch.randn(1, *self.weight_shape))
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
            if self.channel_wise: # cwc
                self.weight_shape = (self.num_weights, self.in1_channels, self.out_channels)
            elif self.connected: # fc
                self.weight_shape = (self.num_weights, self.in1_channels, self.in2_channels, self.out_channels)
            else: # pw
                self.weight_shape = (self.num_weights, self.in1_channels, self.in2_channels)

            if not self.external_weights:
                self.weight = torch.randn(1, *(self.weight_shape))
                if self.channel_wise: # cwc
                    self.weight = self.weight.mul_(sqrt(2/(self.in1_channels + self.out_channels)))
                elif self.connected: # fc
                    self.weight = self.weight.mul_(sqrt(2/((self.in1_channels * self.in2_channels) + self.out_channels)))
                else: # pw
                    self.weight = self.weight.mul_(sqrt(2/(self.in1_channels * self.in2_channels)))
                self.weight = nn.Parameter(self.weight)
    def forward(self, x1: Tensor, x2: Tensor, weight: Optional[Tensor] = None):
        r"""
        Performs the weighted tensor product operation.

        Parameters
        ----------
        x1 : :obj:`~torch.Tensor`
            The first input tensor of shape :math:`(N,\text{num_orders_1},C_1)`.
        x2 : :obj:`~torch.Tensor`
            The second input tensor of shape :math:`(N,\text{num_orders_2},C_2)`. 
        weight : Optional[Tensor], optional
            The weight of shape :math:`(N,\text{num_weights},...)`, where ":math:`...`" 
            depends on the tensor product type as listed above. 
            Default is :obj:`None`.

            It will be used if :obj:`external_weights` is True
            or if provided even when :obj:`external_weights` is False.
        
            
        Returns
        -------
        :obj:`~torch.Tensor`
            The tensor product of shape :math:`(N,\text{num_orders_out},C_{\text{out}})`,
            where :math:`C_{\text{out}}` will be :math:`C=C_1=C_2` if :obj:`tp_type='cw'`,
            :math:`(C_1,C_2)` if :obj:`tp_type='pw'` or the specified value otherwise.
        
            
        Where :math:`N` is the batch-size that will automatically broadcast if set to :math:`1` 
        and :math:`C` is the corresponding number of channels.  

        
        """
        if weight is None and not self.external_weights:
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
        return f'{self.__class__.__name__}(\n  L_in1={self.L_in1}, L_in2={self.L_in2}, L_out={self.L_out}, \n  {type_repr}, external_weights={self.external_weights}\n)'

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
    r""" The traditional tensor product with no weights

    - if :obj:`channel_wise`:

    .. math:: 
        (\mathbf{x}\otimes\mathbf{y})_c = \bigoplus_{l\in L_{\text{out}}}\sum_{l_1,l_2}(\mathbf{x}^{(l_1)}_c\otimes \mathbf{y}^{(l_2)}_c)^{(l)}

    - otherwise (pair-wise):

    .. math:: 
        (\mathbf{x}\otimes\mathbf{y})_{c_1,c_2} = \bigoplus_{l\in L_{\text{out}}}\sum_{l_1,l_2}(\mathbf{x}^{(l_1)}_{c_1}\otimes \mathbf{y}^{(l_2)}_{c_2})^{(l)}

    and

    .. math::
        (\mathbf{x}^{(l_1)}_{c}\otimes \mathbf{y}^{(l_2)}_{c'})^{(l)}_m=\sum_{m_1=-l_1}^{l_1}\sum_{m_2=-l_2}^{l_2}C_{(l_1,m_1)(l_2,m_2)}^{(l,m)}
        \mathbf{x}^{(l_1)}_{m_1,c}\mathbf{y}^{(l_2)}_{m_2,c'}, m=-l,\dots,l,

    where the summation of :math:`(l_1,l_2)` is over all the values such that 
    :math:`l_1\in L_1, l_2\in L_2` and :math:`|l_1-l_2|\le l\le l_1+l_2` and 
    :math:`C_{(l_1,m_1)(l_2,m_2)}^{(l,m)}` are the Clebsch-Gordan coefficients.

    The equivariance of tensor product means that for any rotation 
    :math:`\mathbf{R}\in\text{SO(3)}` and corresponding Wigner D matrices
    :math:`\mathbf{D}_1`, :math:`\mathbf{D}_2` and :math:`\mathbf{D}` on 
    the two input spaces and the output space, correspondingly, it holds

    .. math::
        [(\mathbf{D}_1\mathbf{x})\otimes(\mathbf{D}_2\mathbf{y})]=\mathbf{D}(\mathbf{x}\otimes\mathbf{y}).

    Parameters
    ----------
    L_in1 : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the first input.
    L_in2 : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the second input.
    L_out : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the output.
    channel_wise : bool
        Whether the product is performed channel-wise.
    """
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

    def forward(self, x1: Tensor, x2: Tensor):
        r"""Perform tensor product

        Parameters
        ----------
        x1 : :obj:`~torch.Tensor`
            The first input tensor of shape :math:`(N,\text{num_orders_1},C_1)`
        x2 : :obj:`~torch.Tensor`
            The second input tensor of shape :math:`(N,\text{num_orders_2},C_2)`

        Returns
        -------
        :obj:`~torch.Tensor`
            The tensor product of two input tensors 
            of shape :math:`(N,\text{num_orders_out},C)` if :obj:`channel_wise` is :obj:`True`
            or :math:`(N,\text{num_orders_out},C_1, C_2)` otherwise.


        Where :math:`N` is the batch-size that will automatically broadcast if set to :math:`1` 
        and :math:`C_1,C_2,C` are corresponding number of channels.
        If :obj:`channel_wise` is :obj:`True`, :math:`C_1=C_2=C` should be satisfied.  
        """
        if self.channel_wise:
            return _tensor_product_cw(x1, x2, self.CG_vals, self.M1, self.M2, self.M_ptr)
        else:
            return _tensor_product_pw(x1, x2, self.CG_vals, self.M1, self.M2, self.M_ptr)
        
    def __repr__(self):
        return f'{self.__class__.__name__}(L_in1={self.L_in1}, L_in2={self.L_in2}, L_out={self.L_out}, channel_wise={self.channel_wise})'

class TensorDot(nn.Module):
    r"""
    The module that computes the degree-wise dot product between spherical features

    .. math::

        d_c^{(l)} = \sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_c [{\mathbf{y}_m^{(l)}}]_c

    if :obj:`channel_wise`, or

    .. math::

        d_{c_1,c_2}^{(l)} = \sum_{m=-l}^l [{\mathbf{x}_m^{(l)}}]_{c_1} [{\mathbf{y}_m^{(l)}}]_{c_2}
        
    otherwise.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of inputs.
    channel_wise : bool, optional
        If True, compute channel-wise dot product. Default is :obj:`True`.

    Example
    --------
    >>> L = DegreeRange(0, 3)
    >>> tensor_dot = TensorDot(L, channel_wise=True)
    >>> x1 = torch.randn(32, 16, 64)  # (N, num_orders, C1)
    >>> x2 = torch.randn(32, 16, 64)  # (N, num_orders, C2)
    >>> result = tensor_dot(x1, x2)
    >>> print(result.shape)
    torch.Size([32, 4, 64])  # (N, num_degrees, C)

    """
    def __init__(self, 
                 L: DegreeRange,
                 channel_wise: bool = True
                ):
        super().__init__()
        self.L = L
        self.channel_wise = channel_wise

    def forward(self, x1: Tensor, x2: Tensor):
        r"""
        Compute the degree-wise dot product between two input tensors.

        Parameters
        ----------
        x1 : :obj:`~torch.Tensor`
            First input tensor of shape :math:`(N, \text{num_orders_1}, C_1)`.
        x2 : :obj:`~torch.Tensor`
            Second input tensor of shape :math:`(N, \text{num_orders_1}, C_2)`.

        Returns
        -------
        :obj:`~torch.Tensor`
            The result of the dot product of shape 
            :math:`(N, \text{num_degrees}, C)` if :obj:`channel_wise` is :obj:`True`
            or :math:`(N, \text{num_degrees}, C_1, C_2)` if :obj:`channel_wise` is :obj:`False`.

            
        Where :math:`N` is the batch-size that will automatically broadcast if set to :math:`1`
        and :math:`C_1,C_2,C` are corresponding number of channels.
        If :obj:`channel_wise` is :obj:`True`, :math:`C_1=C_2=C` should be satisfied.  
        """
        return dot(x1, x2, self.L, self.channel_wise)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(L={self.L}, channel_wise={self.channel_wise})'
