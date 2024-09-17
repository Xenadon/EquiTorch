import torch
from ..typing import DegreeRange
from ._clebsch_gordan import blocked_CG

def so3_weights_to_so2(weight_so3: torch.Tensor, L_in:DegreeRange, L_out:DegreeRange, channel_wise=True):
    """
    Transform the weights used for `SO3Linear` to the weights for `SO2Linear`.

    :param weight_so3: The weights used for `SO3Linear`. Shape depends on `channel_wise`:
        - If not channel_wise: `(N, num_degree_in, in_channels, out_channels)` or `(num_degree_in, in_channels, out_channels)`
        - If channel_wise: `(N, num_degree_in, channels)` or `(num_degree_in, channels)`
        It should be sorted by `(l_out, l_in, l_edge)` lexicographically in the `num_degree_in` dimension.
        `l_edge` should take all values from `|l_out-l_in|` to `l_out+l_in` for a given pair of `l_out` and `l_in`.
    :type weight_so3: torch.Tensor
    :param L_in: The degree range of input features
    :type L_in: DegreeRange
    :param L_out: The degree range of output features
    :type L_out: DegreeRange
    :param channel_wise: Whether the weights are channel-wise, defaults to True
    :type channel_wise: bool, optional

    :return: weight_so2: Transformed weights for `SO2Linear`, with the same shape as `weight_so3`,
        sorted by `(l_out, l_in, m)` lexicographically in the `num_degree_in` dimension.
    :rtype: torch.Tensor
    """
    if channel_wise:
        weight_so3 = weight_so3.unsqueeze(dim=-1)
    shared_weight = weight_so3.ndim==3
    weight_so2 = weight_so3.clone()
    offset = 0
    CGs = blocked_CG(L_out, L_in, L_in[1]+L_out[1], dtype=weight_so3.dtype)
    ys = torch.sqrt((2*torch.arange(0, L_in[1]+L_out[1]+1, dtype=weight_so3.dtype)+1) / 4 / torch.pi)
    for l_out in range(L_out[0], L_out[1]+1):
        for l_in in range(L_in[0], L_in[1]+1):
            l_ = min(l_in, l_out)
            transform = torch.zeros(2*l_+1,2*l_+1, dtype=weight_so3.dtype, device=weight_so3.device)
            for ind_l, l  in enumerate(range(l_in+l_out-2*l_, l_in+l_out+1)):
                for ind_m, m in enumerate(range(-l_,l_+1)):
                    transform[ind_m, ind_l] = CGs[l_out, l_in, l][l_out-abs(m), l_in-m, l] * ys[l]
            if not shared_weight:
                transform = transform.unsqueeze(0)
                weight_so2[:,offset:offset+2*l_+1,:,:] = (transform @ weight_so3[:,offset:offset+2*l_+1,:,:].flatten(-2,-1)).unflatten(-1, (weight_so3.shape[-2:]))
            else:
                weight_so2[offset:offset+2*l_+1,:,:] = (transform @ weight_so3[offset:offset+2*l_+1,:,:].flatten(-2,-1)).unflatten(-1, (weight_so3.shape[-2:]))
            offset += 2*l_+1
    return weight_so2.squeeze(dim=-1) if channel_wise else weight_so2

def so2_weights_to_so3(weight_so2: torch.Tensor, L_in:DegreeRange, L_out:DegreeRange, channel_wise=True):
    """
    Transform the weights used for `SO2Linear` to the weights for `SO3Linear`.

    :param weight_so2: The weights used for `SO2Linear`. Shape depends on `channel_wise`:
        - If not channel_wise: `(N, num_degree_in, in_channels, out_channels)` or `(num_degree_in, in_channels, out_channels)`
        - If channel_wise: `(N, num_degree_in, channels)` or `(num_degree_in, channels)`
        It should be sorted by `(l_out, l_in, m)` lexicographically in the `num_degree_in` dimension. 
    :type weight_so2: torch.Tensor
    :param L_in: The degree range of input features
    :type L_in: DegreeRange
    :param L_out: The degree range of output features
    :type L_out: DegreeRange
    :param channel_wise: Whether the weights are channel-wise, defaults to True
    :type channel_wise: bool, optional

    :return: weight_so2: Transformed weights for `SO3Linear`, with the same shape as `weight_so2`,
        sorted by `(l_out, l_in, l_edge)` lexicographically in the `num_degree_in` dimension.
        `l_edge` will take all values from `|l_out-l_in|` to `l_out+l_in` for a given pair of `l_out` and `l_in`.
    :rtype: torch.Tensor
    """
    if channel_wise:
        weight_so2 = weight_so2.unsqueeze(dim=-1)
    shared_weight = weight_so2.ndim==3
    weight_so3 = weight_so2.clone()
    offset = 0
    CGs = blocked_CG(L_out, L_in, L_in[1]+L_out[1], dtype=weight_so2.dtype)
    ys = torch.sqrt((2*torch.arange(0, L_in[1]+L_out[1]+1, dtype=weight_so2.dtype)+1) / 4 / torch.pi)
    for l_out in range(L_out[0], L_out[1]+1):
        for l_in in range(L_in[0], L_in[1]+1):
            l_ = min(l_in, l_out)
            transform_inv = torch.zeros(2*l_+1,2*l_+1, dtype=weight_so2.dtype, device=weight_so2.device)
            for ind_l, l  in enumerate(range(l_in+l_out-2*l_, l_in+l_out+1)):
                for ind_m, m in enumerate(range(-l_,l_+1)):
                    transform_inv[ind_m, ind_l] = CGs[l_out, l_in, l][l_out-abs(m), l_in-m, l] * ys[l]
            transform = transform_inv.inverse()
            if not shared_weight:
                transform = transform.unsqueeze(0)
                weight_so3[:,offset:offset+2*l_+1,:,:] = (transform @ weight_so2[:,offset:offset+2*l_+1,:,:].flatten(-2,-1)).unflatten(-1, (weight_so2.shape[-2:]))
            else:
                weight_so3[offset:offset+2*l_+1,:,:] = (transform @ weight_so2[offset:offset+2*l_+1,:,:].flatten(-2,-1)).unflatten(-1, (weight_so2.shape[-2:]))
            offset += 2*l_+1
    return weight_so3.squeeze(dim=-1) if channel_wise else weight_so3
