import timeit
import torch
import torch_geometric
import e3nn
import torch_scatter

from typing import Tuple, List

# import torch_geometric.utils

import sys

sys.path.append('..')

from equitorch.nn._linear import SO2Linear, SO3Linear
from equitorch.utils._clebsch_gordan import blocked_CG, coo_CG, dense_CG
from equitorch.utils._geometries import align_to_z_wigner, rot_on
from equitorch.math import _o3
from equitorch.utils._indices import degree_order_to_index, degrees_in_range, extract_batch_ptr, list_degrees, order_batch, num_order_between, order_ptr, reduce_order_to_degree
from equitorch.utils._other import so3_weights_to_so2

float_type = torch.float

def SO3Conv(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
    c = SO3Linear(in_channels, out_channels, 
                  (l1_min, l1_max), (l2_min, l2_max), (l_min, l_max), 
                  True, False).to(device)
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, **kwargs):
        return c.forward(X, Y, W)
    return conv

def SO3ContractOnce(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
    # ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    Cs, Ms, ls_cg, l_ind_cg = coo_CG((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    # Cs, Ms, ls_cg, l_ind_cg = coo_CG(CGs.keys(), CGs, l_min, l1_min, l2_min)
    M_batch, M_ptr, M = extract_batch_ptr([Ms[0]])
    Ms = Ms.to(device)
    M_ptr = M_ptr.to(device)
    l_ind_cg = l_ind_cg.to(device)
    Cs = Cs.to(device)
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, **kwargs):
        '''
        X: B * M1 * C
        Y: B * M2
        W: B * Ls * C * C'
        '''
        X = X.index_select(-2, Ms[1]) # B * MM1M2 * C
        Y = Y.index_select(-1, Ms[2]) # B * MM1M2
        # C_val: MM1M2
        inter = (Cs.unsqueeze(0) * Y).unsqueeze(-1) * X # B * MM1M2 * C
        W = W.index_select(dim=-3,index=l_ind_cg) # B * MM1M2 * C * C'
        ret = (inter.unsqueeze(-2) @ W).squeeze(-2) # B * MM1M2 * C'
        ret = torch_geometric.utils.segment(ret, M_ptr.unsqueeze(0)) # B * M * C'
        return ret
    
    return conv

def expand_weight(weight, weight_to, Ml1l2s, l_ind):
    weight_to[:,*Ml1l2s,:,:] = weight[:,l_ind,:,:]
    return weight_to

def generate_lMl1l2s(ls,l_min=0):
    '''
    returns:  [ (l,M,l1,l2), 
                l_ind) ]
    '''
    ret = sorted([(l,degree_order_to_index(l,m,l_min),l1,l2, l_ind) 
                for l_ind, (l, l1, l2) in enumerate(ls) 
                for m in range(-l,l+1)])
    ls_lMl1l2 = torch.tensor([[t[0], t[2], t[3]] for t in ret]).T
    Ms = torch.tensor([t[1] for t in ret])
    l_ind = torch.tensor([t[4] for t in ret])
    return ls_lMl1l2, Ms, l_ind 

def SO3Einsum(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):

    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    ls_lMl1l2, Ms, l_ind = generate_lMl1l2s(ls, l_min)
    l1 = ls_lMl1l2[1]-l1_min
    l2 = ls_lMl1l2[2]-l2_min
    CG = dense_CG((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max), dtype=float_type).to(device)
    l1_ptr = order_ptr((l1_min, l1_max)).to(device)
    l2_ptr = order_ptr((l2_min, l2_max)).to(device)
    shape_W = (num_order_between(l_min, l_max), l1_max-l1_min+1, l2_max-l2_min+1)
    l_ind = l_ind.to(device)
    Ms = Ms.to(device)
    l1 = l1.to(device)
    l2 = l2.to(device)
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, **kwargs):
        '''
        X: B * M1 * C
        Y: B * M2
        W: B * Ls * C * C'
        '''
        non_contract = torch.einsum('PQR,bQC,bR->bPQRC', CG, X, Y)
        inter = torch_geometric.utils.segment(non_contract, l2_ptr.reshape(1,1,1,-1))
        inter = torch_geometric.utils.segment(inter, l1_ptr.reshape(1,1,-1))
        weight = torch.zeros(W.shape[0],*shape_W,W.shape[-2],W.shape[-1],dtype=float_type,device=device)
        weight = expand_weight(W, weight, (Ms,l1,l2), l_ind)
        return torch.einsum('bMpqCD, bMpqC -> bMD',weight,inter)
    
    return conv

def make_basis(Y, CG, l2_min, l2_max):
    # CG: M * M1 * M2
    # idx = order_to_degree_index(l2_min, l2_max, device=Y.device)
    ptr = order_ptr((l2_min, l2_max), dim=3, device=Y.device)
    # torch.einsum('PQR, NR -> NPQR')
    non_contract = Y.reshape(Y.shape[0], 1, 1, Y.shape[1]) * \
        CG.unsqueeze(0)
    return torch_geometric.utils.segment(non_contract, ptr)
    
def _feat_basis_contract(feature, basis, l1_min, l1_max):
    '''
        feature: N * M_i * C_i
        basis: N * M_o * M_i * L_e 
        order_batch: M_i, in range(0, L_i)

        out: N * M_o * L_i * L_e * C_o
        complexity: O(NCL^5)
    '''
    N, Mout, L_e, Min = basis.shape
    non_contract_feat_basis = feature.unsqueeze(1).unsqueeze(-2) * basis.unsqueeze(-1)
    ret = reduce_order_to_degree(non_contract_feat_basis, (l1_min, l1_max), dim=2)
    return ret

def _feat_basis_transform(feature_basis, weight, l_min, l_max):
    '''
        feat_basis: N * M_o * L_i * L_e * C_i
        weight: N * M_o * L_i * L_e * C_i * C_o
        m2l_ind: M_o, in range(0, L_o)
    '''
    N, M_o, L_e, L_i, i = feature_basis.shape
    # expanded_weight = expand_degree_to_order(weight, *L_out, dim=1).view(N * M_o, L_e * L_i * i, -1)
    return (feature_basis.view(N * M_o, 1, -1) @ 
            weight.view(N*M_o, -1, weight.shape[-1])).view(N, M_o, -1)

def SO3CachedEinsum(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    ls_lMl1l2, Ms, l_ind = generate_lMl1l2s(ls, l_min)
    l1 = ls_lMl1l2[1]-l1_min
    l2 = ls_lMl1l2[2]-l2_min
    CG = dense_CG((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max), dtype=float_type).to(device)
    Ms = Ms.to(device)
    l1 = l1.to(device)
    l2 = l2.to(device)
    l_ind = l_ind.to(device)
    shape_W = (num_order_between(l_min, l_max), l1_max-l1_min+1, l2_max-l2_min+1)
    basis = []
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, clear_basis=False, **kwargs):
        '''
        X: B * M1 * C
        Y: B * M2
        W: B * Ls * C * C'
        '''
        if len(basis) == 0 or clear_basis:
            b = make_basis(Y, CG, l2_min, l2_max)
            basis.clear()
            basis.append(b)
        else:
            b = basis[-1]

        fb = _feat_basis_contract(X, b, l1_min, l1_max)

        weight = torch.zeros(W.shape[0],*shape_W,W.shape[-2],W.shape[-1],dtype=float_type, device=device)
        weight = expand_weight(W, weight, (Ms,l1,l2), l_ind)
        return _feat_basis_transform(fb, weight, l_min, l_max)
    
    return conv

def SO3e3nn(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device):
    rep_in1 = '+'.join(f'{in_channels}x{l}e' for l in range(l1_min, l1_max+1))
    rep_in2 = '+'.join(f'1x{l}o' for l in range(l2_min, l2_max+1))
    rep_out = '+'.join(f'{out_channels}x{l}o' for l in range(l_min, l_max+1))
    tp = e3nn.o3.FullyConnectedTensorProduct(
        rep_in1, rep_in2, rep_out, shared_weights=False)
    tp.weight = tp.weight.type(float_type)
    tp.to(device)
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, **kwargs):
        '''
        X: B * C * M1
        Y: B * M2
        W: B * Ls * C * C'
        '''
        ret = tp.forward(X.flatten(-2,-1), Y, W.flatten(start_dim=1))
        return ret
    return conv

def SO2Conv(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device):
    c = SO2Linear(in_channels, out_channels, 
                  (l1_min, l1_max), (l_min, l_max),  
                  True, False).to(device)
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, D_in:torch.Tensor, D_out:torch.Tensor, **kwargs):
        return rot_on(D_out.transpose(-1,-2),c.forward(rot_on(D_in, X), W))
    return conv

Convs = [
    SO3Conv, 
    SO3ContractOnce, 
    SO3CachedEinsum, 
    SO3Einsum,
    SO3e3nn,
    SO2Conv
]

def init_test_perf(N, in_channels, out_channels, L_in, L_out, len_W, device='cpu'):
    L_edge = (min([abs(l1_-l_) for l1_ in degrees_in_range(L_in) 
                   for l_ in degrees_in_range(L_out)]), L_out[1]+L_out[1])
    r = torch.randn(N,3,dtype=float_type)
    r = r / r.norm(dim=-1, keepdim=True)
    X = torch.randn(N,num_order_between(*L_in),in_channels)
    Y = _o3.spherical_harmonics(r, L_edge)
    W_so3 = torch.randn(N, len_W,in_channels, out_channels, dtype=float_type)
    W_so2 = so3_weights_to_so2(W_so3, L_in, L_out, channel_wise=False)
    D_align_in = align_to_z_wigner(r, L_in)
    D_align_out = align_to_z_wigner(r, L_out)
    return (
        X.to(device), Y.to(device), 
        W_so3.to(device), W_so2.to(device),  
        D_align_in.to(device), D_align_out.to(device)
    )

def time_forward(i, N, L_in, L_out, in_channels, out_channels):
    l1_min, l1_max = L_in
    l_min, l_max = L_out
    L_edge = (min([abs(l1_-l_) for l1_ in degrees_in_range(L_in) 
                for l_ in degrees_in_range(L_out)]), L_out[1]+L_out[1])
    l2_min, l2_max = L_edge
    ls = list_degrees(L_out, L_in, L_edge)
    conv = Convs[i](in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda:1')

    X, Y, W_so3, W_so2, D_in, D_out= init_test_perf(
        N, in_channels, out_channels,
        L_in, L_out, len(ls), device='cuda:1')
    conv(X, Y, W_so3, clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)
    
    ret = {}
    
    try:
        # torch.cuda.empty_cache()
        t = timeit.repeat(
            'conv(X, Y, W_so3, clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)', 
            repeat=repeat, number=number,
            globals = locals()|globals(),
            setup='torch.cuda.empty_cache();\nconv(X, Y, W_so3, clear_basis=True, W_so2=W_so2, D_in=D_in, D_out=D_out);\n'
        )
        
        print(f'conv {i+1}: {min(t)/number*1000} ms')
        ret[i+1] = min(t)/number
    except Exception as e:
        print(f'conv {i+1}: '+str(e.__traceback__.format_exc()))
        ret[i+1] = float('nan')


def time_backward(i, N, L_in, L_out, in_channels, out_channels):

    def test_backward(conv, X, Y, W, **kwargs):
        y = conv(X, Y, W, **kwargs).sum()
        y.backward()

    l1_min, l1_max = L_in
    l_min, l_max = L_out
    L_edge = (min([abs(l1_-l_) for l1_ in degrees_in_range(L_in) 
                for l_ in degrees_in_range(L_out)]), L_out[1]+L_out[1])
    l2_min, l2_max = L_edge
    ls = list_degrees(L_out, L_in, L_edge)
    conv = Convs[i](in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda:1')

    X, Y, W_so3, W_so2, D_in, D_out= init_test_perf(
        N, in_channels, out_channels,
        L_in, L_out, len(ls), device='cuda:1')
    W_so3 = torch.nn.Parameter(W_so3,requires_grad=True)
    W_so2 = torch.nn.Parameter(W_so2,requires_grad=True)
    if W_so3.grad is not None:
        W_so3.grad.zero_()
    if W_so2.grad is not None:
        W_so2.grad.zero_()
    test_backward(conv, X, Y, W_so3, clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)

    ret = {}
    
    t = timeit.repeat(
        'test_backward(conv, X, Y, W_so3, clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)',
        repeat=repeat, number=number,
        globals = locals()|globals(),
        setup='torch.cuda.empty_cache();\ntest_backward(conv, X, Y, W_so3, clear_basis=True, W_so2=W_so2, D_in=D_in, D_out=D_out)'
    )
    print(f'conv {i+1}: {min(t)/number*1000} ms')
    ret[i+1] = min(t)/number


number = 1
repeat = 100