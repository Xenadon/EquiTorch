from math import sqrt
from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing

from .linear import SO2Linear
from ..utils.indices import degrees_in_range, reduce
from ..utils.geometries import rot_on
from ..utils.hooks import get_kwargs_filter_hook

from ..typing import DegreeRange

class SimpleDotAttention(nn.Module):
    r"""
    Computes simple dot attention weights.

    This module computes the attention weights using the dot product of the query and key tensors.

    .. math::
        \alpha_{ij} = \mathrm{softmax}_{j}\left(\frac{\mathbf{q}_{i} \cdot \mathbf{k}_j}{\sqrt{d}}\right)

    where :math:`\alpha` is the attention weight, :math:`\mathbf{q}` is the query tensor, 
    :math:`\mathbf{k}` is the key tensor, and :math:`d` is the channel numbers of the query and key tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, qk: Tuple[Tensor, Tensor],
                index: Optional[Tensor] = None, 
                ptr: Optional[Tensor] = None, 
                num_nodes: Optional[int] = None):
        r"""
        Computes the attention weights.

        Parameters
        ----------
        qk : Tuple[Tensor, Tensor]
            Tuple containing the query tensor :math:`\mathbf{q}` and the key tensor :math:`\mathbf{k}`.
        index: Tensor, optional
            The indices of elements for applying the aggregation. 
            One of :obj:`index` or :obj:`ptr` must be defined.

            Default is :obj:`None`.

        ptr: Tensor, optional
            If given, computes the aggregation based on sorted inputs in CSR representation.
            One of :obj:`index` or :obj:`ptr` must be defined.

            Default is :obj:`None`.

        num_nodes: int, optional
            The size of the output tensor at dimension dim after aggregation.

        Returns
        -------
        :obj:`~torch.Tensor`
            Attention weights.

        """
        q, k = qk
        pre_softmax = (q*k).sum((-1,-3)) * (q.shape[-1] ** -0.5)
        return softmax(pre_softmax, index = index, ptr = ptr, num_nodes=num_nodes, dim=0)

class SE3TrAttention(nn.Module):  # only computes the attention weights
    r'''
    The attention score used in SE(3)-Transformer.

    .. math ::
        \begin{aligned}
        \mathbf{q}_{ij} &= \tilde{\mathbf{W}}_Q\mathbf{x}_{i}, \\
        \mathbf{k}_{ij} &= \tilde{\mathbf{W}}_K(\|\mathbf{r}_{ij}\|)\mathbf{x}_{j}, \\
        {z}_{ij} &= \frac{1}{\sqrt{\text{k_channels}}}\sum_{l,m,c}(\mathbf{q}_{ij})_{m,c}^{(l)}(\mathbf{k}_{ij})_{m,c}^{(l)}\\
        \alpha_{ij} &= \mathop{\mathrm{softmax\ }}_{j\in\mathcal{N}(i)}{z}_{ij}\\
                   &= \frac{\exp({z}_{ij})}{\displaystyle \sum_{j'\in\mathcal{N}(i)}\exp({z}_{ij'})},        \\
        \end{aligned}     

    where :math:`\tilde{\mathbf{W}}_{\phi;Q}, \tilde{\mathbf{W}}_{\phi;K}(\|\mathbf{r}_{ij}\|)`
    are two :obj:`SO2Linear` operations. The weight of :math:`\tilde{\mathbf{W}}_{\phi;Q}` does not 
    depend on the edge vectors.

    This computes invariant attention score using scaled dot product attention as the attention 
    score (11) in `SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks
    <https://arxiv.org/abs/2006.10503>`_ except that we use the more efficient 
    :obj:`SO2Linear` here instead of :obj:`SO3Linear` and the scaling for the dot products.


    For more details on how the softmax is computed, see `softmax <https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.softmax>`_.


    Parameters
    ----------
    L_in : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the input.
    L_k : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the keys and values.
    in_channels : int
        The number of input channels.
    k_channels : int
        The number of key and value channels.
    num_heads : int, optional
        The number of heads, Default is 1.
    '''
    def __init__(
            self,
            L_in: DegreeRange,
            L_k: DegreeRange,
            in_channels: int,
            k_channels: int,
            num_heads: int = 1,
            weight_k_producer: Optional[Callable] = None,
    ):
        super().__init__()
        self.L_in = L_in
        self.L_k = L_k
        self.in_channels = in_channels
        self.k_channels = k_channels
        self.num_heads = num_heads
        self.proj_k = SO2Linear(L_in, L_k, in_channels, num_heads*k_channels, weight_k_producer is not None, False)
        self.proj_q = SO2Linear(L_in, L_k, in_channels, num_heads*k_channels, False, False)
        self.weight_k_producer = weight_k_producer
        self.weight_k_shape = (-1, self.proj_k.num_weights, in_channels, num_heads*k_channels)

        self.scale = (self.k_channels//self.num_heads) ** (-0.5)

    def forward(self, qk: Tuple[Tensor, Tensor],
                edge_emb: Optional[Tensor] = None,
                index: Optional[Tensor] = None, 
                ptr: Optional[Tensor] = None, 
                num_nodes: Optional[int] = None):
        r"""forward 
        """
        x_q, x_k = qk
        q = self.proj_q(x_q).unflatten(-1, (self.num_heads, self.k_channels))
        k = self.proj_k(
            x_k, self.weight_k_producer(edge_emb).reshape(self.weight_k_shape)).unflatten(
                -1, (self.num_heads, self.k_channels)) 
        pre_softmax = (q*k).sum((-1,-3)) * self.scale
        return softmax(pre_softmax, index = index, ptr = ptr, num_nodes=num_nodes, dim=0)

class MultiheadAttentionBlock(nn.Module):
    r"""
    Computes the result of multihead attention

    .. math::
        \mathbf{m}_{ij}=\mathop{\Large\oplus}_{h}\alpha_{h;ij}\mathbf{v}_{h;ij},

    where 

    .. math::
        \begin{aligned}
        \alpha_{h;ij}&=\text{attention_score_producer}(\mathbf{qk}_{ij};...),\\
        \mathbf{v}_{h;ij}&=\text{v_producer}(\tilde{\mathbf{v}}_{ij},...),
        \end{aligned}

    :math:`\mathbf{qk}_{ij}` can be a tuple of separated tensors :math:`(\mathbf{q}_i,\mathbf{k}_j)`
    or a fused (say, concated or added) tensor :math:`(\mathbf{qk})_{ij}`.
    and ":math:`...`" means any additional arguments that passed in.

    Parameters
    ----------
    num_heads : int, optional
        Number of attention heads. Default is 1.
    attention_score_producer : Callable, optional
        Produces the attention scores for the multi-head attention. 
        It should at least take an input :obj:`qk` and the index :obj:`index`
        denoting the independent parts for normalization or the corresponding 
        :obj:`ptr`, but it can take more inputs.
        
        The returned shape should be :math:`(...,e,h)` for :math:`e`
        representing :math:`i,j` pairs and :math:`h` representing head dimensions.
        
        Default is :obj:`SimpleDotAttention` if not provided.
    v_producer : Callable, optional
        Produces the value tensor for the multi-head attention mechanism. 
        It should at least take an input :obj:`v` but can take more inputs.

        The returned number of channels should be :math:`C_v\cdot h` for all
        :math:`i,j` pairs, where :math:`C_v` representing the value channels
        of one head and :math:`h` representing head dimensions. When use sum 
        or mean head aggregations, :math:`C_v` should be exactly desired number of 
        output channels but for concat aggregation :math:`C\cdot h` together should
        be the desired number of output channels.  

        Defaults to Identity if not provided.

    head_aggr : str, optional
        Specifies how the outputs from different attention heads should be aggregated. 
        Can be one of 'sum', 'add', 'mean', 'avg', 'cat', or 'concat'. Defaults to 'cat'.
    """
    
    def __init__(
            self,
            num_heads: int = 1,
            attention_score_producer: Optional[Callable] = None,
            v_producer: Optional[Callable] = None,
            head_aggr: str = 'cat', # 'sum'/'add', 'mean'/'avg', 'cat'/'concat' 
            need_weights: bool = False
    ):
        super().__init__()

        self.num_heads = num_heads

        self.need_weights = need_weights

        self.head_aggr = head_aggr

        if self.head_aggr == 'sum' or self.head_aggr == 'add':
            self.head_aggr_fn = lambda x: x.sum(dim=-2)
        elif self.head_aggr == 'mean' or self.head_aggr == 'avg':
            self.head_aggr_fn = lambda x: x.mean(dim=-2)
        elif self.head_aggr == 'cat' or self.head_aggr == 'concat':
            self.head_aggr_fn = lambda x: x.flatten(-2, -1)
        else:
            raise ValueError("Invalid value for head_aggr. Must be one of 'sum', 'add', 'mean', 'avg', 'cat', or 'concat'.")

        if attention_score_producer is None:
            self.attention_score_producer = SimpleDotAttention()
        else:
            self.attention_score_producer = attention_score_producer

        if v_producer is None:
            self.v_producer = nn.Identity()
        else:
            self.v_producer = v_producer

        if isinstance(self.attention_score_producer, nn.Module):
            self.attention_score_producer.register_forward_pre_hook(
                get_kwargs_filter_hook(self.attention_score_producer), with_kwargs=True)

        if isinstance(self.v_producer, nn.Module):
            self.v_producer.register_forward_pre_hook(
                get_kwargs_filter_hook(self.v_producer), with_kwargs=True)

    def forward(self, qk: Union[Tensor, Tuple[Tensor, Tensor]], v: Tensor,
                # edge_weight: Optional[Tensor] = None,
                # DT_out: Optional[Tensor] = None, 
                index:Optional[Tensor] = None, 
                ptr: Optional[Tensor] = None,
                num_nodes: Optional[int] = None,
                **kwargs):
        r"""
        Parameters
        ----------
        qk: Union[Tensor, Tuple[Tensor, Tensor]]
            The query-key pairs for each edges. It can be a tuple of separated tensors 
            :math:`(\mathbf{q}_i,\mathbf{k}_j)` or a fused (say, concated
            or added) tensor :math:`\mathbf{qk}_{ij}`.
        v: Tensor
            The tensor to produce values for each edges.
        DT_out: Tensor, optional
            The Wigner D matrix corresponding the rotation that can align the edge vector 
            to the z-axis on the output spaces. If provided, the produced value 
            will be transformed by these matrices before aggregated.
            Default is :obj:`None`.
        edge_weight: Tensor, optional
            The edge weight. If provided, the message :math:`\alpha_{ij}\mathbf{v}_{ij}`
            will be reweighted based on the weights.
            Default is :obj:`None`.
        index: Tensor, optional
            The indices of elements for applying the aggregation. 
            One of :obj:`index` or :obj:`ptr` must be defined.
            Default is :obj:`None`.
        ptr: Tensor, optional
            If given, computes the aggregation based on sorted inputs in CSR representation.
            One of :obj:`index` or :obj:`ptr` must be defined.
            Default is :obj:`None`.
        num_nodes: int, optional
            The size of the output tensor at dimension dim after aggregation.
            Default is :obj:`None`.
        **kwargs:
            Any keyword arguments that will be passed to attention score producer 
            and value producer.
        """
        att_shape = (-1,) + (v.ndim-2) * (1,) + (self.num_heads, 1)
        att = self.attention_score_producer(
            qk, index=index, ptr=ptr, dim_size=num_nodes, **kwargs).view(att_shape)
        
        value = self.v_producer(v, **kwargs).unflatten(-1,(self.num_heads, -1)) 
        
        out = self.head_aggr_fn(att * value)

        return out, att