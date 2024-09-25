from math import sqrt
from typing import Callable, Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing

from ._linear import SO2Linear
from ..utils._indices import degrees_in_range
from ..utils._geometries import rot_on


from ..typing import DegreeRange

class ScaledDotAttention(nn.Module):  # only computes the attention weights
    r'''
    Computes invariant attention score using scaled dot product attention as:

    .. math::
        \begin{aligned}
        \mathbf{q}_{k} &= \tilde{\mathbf{W}}_Q(\|\mathbf{r}\|_Q)\mathbf{x}_{Q,k}, \\
        \mathbf{k}_{k} &= \tilde{\mathbf{W}}_K(\|\mathbf{r}\|_K)\mathbf{x}_{K,k}, \\
        {z}_{k} &= \frac{1}{\sqrt{\text{k_channels}}}\sum_{l,m,c}(\mathbf{q}_{k})_{m,c}^{(l)}(\mathbf{k}_{k})_{m,c}^{(l)}\\
        \alpha_{k} &= \mathop{\mathrm{softmax\ }}_{\text{index}[k]}{z}_k\\
                   &= \frac{\exp({z}_{k})}{\displaystyle \sum_{k':\text{index}[k']=\text{index}[k]}\exp({z}_{k'})},        \\
        \end{aligned}

    where :math:`\tilde{\mathbf{W}}_Q(\|\mathbf{r}\|_Q), \tilde{\mathbf{W}}_K(\|\mathbf{r}\|_K)`
    are two :obj:`SO2Linear` operations that may depend on additional query/key features 
    :math:`\mathbf{r}_Q` and :math:`\mathbf{r}_K`.

    The attention is normalized over all samples :math:`k` with the same index :math:`\text{index}[k]`.

    For more details on how the softmax is computed, see `softmax <https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.softmax>`_.

    In the case where :math:`k` represents the edges :math:`(i,j)`,
    queries are from target nodes :math:`i`, keys are from source nodes 
    :math:`j`, the index is the target index :math:`i`, the query projector is
    edge-independent and the key projector depends on the edge vector 
    :math:`\mathbf{r}_{ij}`, the attention score is computed as

    .. math ::
        \begin{aligned}
        \mathbf{q}_{ij} &= \tilde{\mathbf{W}}_Q\mathbf{x}_{i}, \\
        \mathbf{k}_{ij} &= \tilde{\mathbf{W}}_K(\|\mathbf{r}_{ij}\|)\mathbf{x}_{j}, \\
        {z}_{ij} &= \frac{1}{\sqrt{\text{k_channels}}}\sum_{l,m,c}(\mathbf{q}_{ij})_{m,c}^{(l)}(\mathbf{k}_{ij})_{m,c}^{(l)}\\
        \alpha_{ij} &= \mathop{\mathrm{softmax\ }}_{j\in\mathcal{N}(i)}{z}_{ij}\\
                   &= \frac{\exp({z}_{ij})}{\displaystyle \sum_{j'\in\mathcal{N}(i)}\exp({z}_{ij'})},        \\
        \end{aligned}     

    This is equivalent to the attention score (11) in `SE(3)-Transformers: 
    3D Roto-Translation Equivariant Attention Networks <https://arxiv.org/abs/2006.10503>`_
    except that we use the more efficient :obj:`SO2Linear` here instead of :obj:`SO3Linear`.

    Parameters
    ----------
    L_in : DegreeRange
        The degree range of the input.
    L_k : DegreeRange
        The degree range of the keys and values.
    in_channels : int
        The number of input channels.
    k_channels : int
        The number of key and value channels.
    num_heads : int, optional
        The number of heads, Default is 1.
    weight_k_producer : Callable, optional
        The module to produce weights of the key projector from 
        additional key features. Default is None.
        If not provided, the key projector will not use :obj:`external_weight`.
    weight_q_producer : Callable, optional
        The module to produce weights of the query projector from 
        additional query features. Default is None.
        If not provided, the key projector will not use :obj:`external_weight`.
    '''
    def __init__(
            self,
            L_in: DegreeRange,
            L_k: DegreeRange,
            in_channels: int,
            k_channels: int,
            num_heads: int = 1,
            weight_k_producer: Optional[Callable] = None,
            weight_q_producer: Optional[Callable] = None
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

    def forward(self, x_q:Tensor, x_k:Tensor, q_feat: Optional[Tensor]=None, k_feat: Optional[Tensor]=None, 
                index: Optional[Tensor] = None, ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None):
        r"""forward 

        Parameters
        ----------
        x_q : Tensor
            The query tensor with shape :math:`[K, \text{num_orders}, C]`, where
            :math:`K` is the number of key-query pairs and :math:`C` is the number of channels.
        x_k : Tensor
            The key tensor with shape :math:`[K, \text{num_orders}, C]`, where
            :math:`K` is the number of key-query pairs and :math:`C` is the number of channels.
        q_feat : Tensor, optional
            Additional feature to produce weights for query projector. Default is None
        k_feat : Tensor, optional
            Additional feature to produce weights for key projector. Default is None
        index : Tensor, optional
            The indices of elements for applying the softmax. Default is None
        ptr : Tensor, optional
            If given, computes the softmax based on sorted inputs in CSR representation. Default is None
        num_nodes : int, optional
            The number of nodes. Default is None.

        Returns
        -------
        Tensor
            The attention weights with shape :math:`[K, \text{num_heads}]`
        """
        q = self.proj_q(x_q, self.weight_q_producer(q_feat).reshape(self.weight_q_shape)\
                        if self.weight_q_producer is not None else None).unflatten(-1, self.num_heads, self.k_channels)
        k = self.proj_k(x_k, self.weight_k_producer(k_feat).reshape(self.weight_k_shape)\
                        if self.weight_k_producer is not None else None).unflatten(-1, self.num_heads, self.k_channels) 
        pre_softmax = (q*k).sum((-1,-3)) * (self.k_channels ** -0.5)
        return softmax(pre_softmax, index = index, ptr = ptr, num_nodes=num_nodes, dim=0)

class AttentionalBlock(MessagePassing):

    r"""
    Perform attentional aggregation

    .. math::
        \mathbf{x}'_i=\sum_{j\in\mathcal{N}(i)}\alpha_{ij}\mathbf{v}_j,

    where

    .. math::
        \begin{aligned}
        \alpha_{ij}&=\text{attention_score_producer}(\mathbf{x}_i,\mathbf{x}_j),\\
        \mathbf{v}_j &= \text{v_producer}(\mathbf{x}_j).
        \end{aligned}

    :math:`\alpha` should be SO(3) invariant and :math:`\mathbf{v}` should be
    SO(3) equivariant. Then the whole block will be SO(3) equivariant. 

    Parameters
    ----------
    L_in : DegreeRange
        The degree range of the input.
    L_out : DegreeRange
        The degree range of the output.
    in_channels : int
        The number of input channels.
    k_channels : int
        The number of key and value channels.
    out_channels : int
        The number of output channels.
    num_heads : int, optional
        The number of heads, Default is 1.
    attention_score_producer : Callable, optional
        The module to produce attention scores, Default is None.
        It should take two tensors representing the (q,k) pairs 
        and return the attention score between them. 
        If not provided, the :obj:`ScaledDotAttention` will be used,
    v_producer : Callable, optional
        The module to produce values. Default is None.
        If not provided, the :obj:`SO2Linear` will be used and the 
        :obj:`external_weight` will be True only when `weight_v_producer`
        is provided. 
    weight_q_producer : Callable, optional
        The module to produce weight from edge embeddings for :obj:`attention_score_producer`. 
        This will not be used when the :obj:`attention_score_producer`
        is explicitly passed. 
        Default is None.
    weight_k_producer : Callable, optional
        The module to produce weights from edge embeddings for :obj:`attention_score_producer`. 
        This will not be used when the :obj:`attention_score_producer`
        is explicitly passed. 
        Default is None.
    weight_v_producer : Callable, optional
        The module to produce weights from edge embeddings for :obj:`v_producer`. 
        This will not be used when the :obj:`v_producer`
        is explicitly passed. 
        Default is None.
    """
    def __init__(
            self,
            L_in: DegreeRange,
            L_out: DegreeRange,
            in_channels: int,
            k_channels: int,
            out_channels: int,
            num_heads: int = 1,
            attention_score_producer: Optional[Callable] = None,
            v_producer: Optional[Callable] = None,
            weight_q_producer: Optional[Callable] = None,
            weight_k_producer: Optional[Callable] = None,
            weight_v_producer: Optional[Callable] = None,
    ):
        super().__init__()
        self.L_in = L_in
        self.L_out = L_out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dim_v = out_channels // num_heads
        self.weight_v_producer = weight_v_producer
        self.attention_score_producer = attention_score_producer if attention_score_producer is not None\
            else ScaledDotAttention(L_in, L_in, in_channels, k_channels, num_heads, weight_k_producer, weight_q_producer)
        self.v_producer = v_producer if v_producer is not None\
            else SO2Linear(L_in, L_out, in_channels, out_channels, weight_v_producer is not None, False)

    def forward(self, x: Tensor, edge_index: Tensor, 
                edge_emb: Optional[Tensor] = None, 
                D_in: Optional[Tensor] = None, 
                DT_out: Optional[Tensor] = None, 
                edge_weight: Optional[Tensor] = None
                ):
        """Runs the forward pass of the module

        Parameters
        ----------
        x : torch.Tensor
        edge_index : torch.Tensor
        edge_emb : Tensor, optional
            The edge features that will be used to produce
            weights for Q/K/V projectors, Default is None.
        D_in : Tensor, optional
            The Wigner-D matrix on input space corresponding to the 
            rotation that aligns the edge vector to z axis.
            If provided, the input feature will first be rotated by :obj:`D_in`
            Default is None.
        DT_out : Tensor, optional
            The transpose of Wigner-D matrix on output space corresponding to the 
            rotation that aligns the edge vector to z axis.
            If provided, the value vector of each edge will be rotated by :obj:`D_out` before aggregated.
            Default is None.
        edge_weight: Tensor, optional
            The weight of each edge, used to scale the message along the edges.
        Returns
        -------
        Tensor
        """
        if D_in is not None:
            x = rot_on(D_in, x)
        x1 = self.propagate(edge_index=edge_index, x=x, edge_emb=edge_emb, DT_out=DT_out, edge_weight=edge_weight)
        return x1
    def message(self, x_j: Tensor, x_i: Tensor, edge_index: Tensor,
                edge_emb: Optional[Tensor], DT_out: Optional[Tensor],
                edge_weight: Optional[Tensor]):
        """
        """
        # x_j to x_i, i: query, j: key
        att = self.attention_score_producer(x_i, x_j, edge_emb, edge_emb, edge_index[1]).unsqueeze(-1).unsqueeze(1)
        # att: E * 1 * num_heads * 1
        weight_v = self.weight_v_producer(edge_emb) if self.weight_v_producer is not None else None
        v = self.v_producer.forward(x_j, weight_v).unflatten(-1, (self.num_heads, self.dim_v))
        # E * M * num_heads * dim_v
        out = (att * v).flatten(-2, -1)

        if DT_out is not None:
            out = rot_on(DT_out, out)
        if edge_weight is not None:
            out = edge_weight.view(-1,1,1) * out

        return out