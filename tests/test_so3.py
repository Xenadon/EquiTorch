import timeit
import torch
import torch_geometric
import e3nn
import torch_scatter

from typing import Tuple, List

# import torch_geometric.utils

import sys

from equitorch.utils._weights import so3_weights_to_so2
sys.path.append('..')

from equitorch.nn._linear import SO2Linear, SO3Linear
from equitorch.utils._clebsch_gordan import blocked_CG, coo_CG, dense_CG
from equitorch.utils._geometries import align_to_z_wigner, rot_on
from equitorch.math import _o3
# from equitorch.utils._clebsch_gordan import partial_tensor_product
from equitorch.utils._indices import degree_order_to_index, extract_batch_ptr, list_degrees, order_batch, num_order_between, order_ptr, reduce_order_to_degree

float_type = torch.float

def SO3Conv1(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
    c = SO3Linear(in_channels, out_channels, 
                  (l1_min, l1_max), (l2_min, l2_max), (l_min, l_max), 
                  True, False).to(device)
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, **kwargs):
        return c.forward(X, Y, W)
    return conv


def SO3Conv2(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
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

def SO3Conv3(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
    '''
    Not optimized, maybe unfair for performance
    '''
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    # weight = torch.randn(len(ls), out_channels, in_channels)
    # CG = dense_CG(l_min, l_max, l1_min, l1_max, l2_min, l2_max)

    CGs = {k:v.to(device) for k,v in blocked_CG((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max), dtype=float_type).items()}

    # coo = coo_CGs(ls, CGs)
    # (l_cg, M_cg, l1_cg, l2_cg, M1_cg, M2_cg, l_ind_cg, C_val) = extract_CG_indices(coo)
    # lM_batch, lM_ptr, lM = extract_batch_ptr([t[0] for t in coo], 4)

    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, **kwargs):
        '''
        X: B * M1 * C
        Y: B * M2
        W: B * Ls * C * C'
        '''
        ret = torch.zeros(X.shape[0], (l_max+1)**2-l_min**2, W.shape[-1], dtype=float_type, device=device)
        for i,(l,l1,l2) in enumerate(CGs.keys()):
            ret[:,l**2-l_min**2:(l+1)**2-l_min**2, :] += \
                torch.einsum('BCD, pqr, BqC, Br -> BpD', 
                             W[:,i,:,:], CGs[l,l1,l2], 
                             X[:,l1**2-l1_min**2:(l1+1)**2-l1_min**2,:], 
                             Y[:,l2**2-l2_min**2:(l2+1)**2-l2_min**2])
        return ret
    
    return conv

def expand_weight(weight, weight_to, Ml1l2s, l_ind):
    weight_to[:,*Ml1l2s,:,:] = weight[:,l_ind,:,:]
    return weight_to

# def dense_CG(l_min, l_max, l1_min, l1_max, l2_min, l2_max, condition=None):
#     CGs = [
#         [
#             [
#                 e3nn.o3._wigner._so3_clebsch_gordan(l,l1,l2).nan_to_num(0).type(float_type)
#             for l2 in range(l2_min, l2_max+1)]
#         for l1 in range(l1_min,l1_max+1)]
#     for l in range(l_min,l_max+1)]
#     CG = torch.cat(
#         [torch.cat(
#             [torch.cat(
#                 CGs_LL1, dim=-1) 
#             for CGs_LL1 in CGs_L], dim=-2
#         ) for CGs_L in CGs], dim=-3)
#     return CG

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

def SO3Conv4(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
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


def SO3Conv5(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cpu'):
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

# def e3nn_prepare(X, Y, in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max)

def SO3Conv6(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device):
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

def init_test_perf(N, in_channels, out_channels, l_l, l1_l, l2_l, len_W, device='cpu'):
    r = torch.randn(N,3,dtype=float_type)
    r = r / r.norm(dim=-1, keepdim=True)
    X = torch.randn(N,num_order_between(l1_l[0], l1_l[-1]),in_channels)
    Y = _o3.spherical_harmonics(r, (l2_l[0], l2_l[-1]))
    W = torch.randn(N, len_W,in_channels, out_channels, dtype=float_type)
    W_so2 = so3_weights_to_so2(W, (l1_l[0], l1_l[-1]), (l_l[0], l_l[-1]),)
    D_align_in = align_to_z_wigner(r, (l1_l[0], l1_l[-1]))
    D_align_out = align_to_z_wigner(r, (l_l[0], l_l[-1]))
    # W = torch.ones(N, len_W,in_channels, out_channels, dtype=float_type)
    return W.to(device), X.to(device), Y.to(device), W_so2.to(device), D_align_in.to(device), D_align_out.to(device)

number = 1
repeat = 100

Convs = [SO3Conv1, SO3Conv2, SO3Conv3, SO3Conv4, SO3Conv5, SO3Conv6, SO2Conv]

def check_forward_single(i, N, L1, L, in_channels, out_channels):
    # print(i+1)
    L2 = (min([abs(l1_-l_) for l1_ in range(L1[0],L1[1]+1) for l_ in range(L[0],L[1]+1)]), L[1]+L1[1])
    l1_min, l1_max = L1
    l2_min, l2_max = L2
    l_min, l_max = L
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    conv = Convs[i](in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda:1')

    W, X, Y, W_so2, D_in, D_out= init_test_perf(
        N, in_channels, out_channels, 
        list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
        len(ls), device='cuda:1')
    conv(X, Y, W, clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)
    
    ret = {}
    
    try:
        # torch.cuda.empty_cache()
        t = timeit.repeat(
            'conv(X, Y, W,clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)', repeat=repeat, number=number,
            # globals={'convs': convs, 'X':X, 'Y':Y, 'W':W, 'i':i, 'torch':torch},
            globals = locals()|globals(),
            # setup='torch.cuda.empty_cache();\nconvs[i](X,Y,W,clear_basis=True)\n'
            setup='torch.cuda.empty_cache();\nconv(X,Y,W,clear_basis=True, W_so2=W_so2, D_in=D_in, D_out=D_out);\n'
        )
        
        print(f'conv {i+1}: {min(t)/number*1000} ms')
        ret[i+1] = min(t)/number
    except Exception as e:
        print(f'conv {i+1}: '+str(e.__traceback__.format_exc()))
        ret[i+1] = float('nan')


def test_backward(conv, W, X, Y, **kwargs):
    # print(W)
    y = conv(X, Y, W, **kwargs).norm()
    y.backward()

def check_backward_single(i, N, L1, L, in_channels, out_channels):
    # print(i+1)
    L2 = (min([abs(l1_-l_) for l1_ in range(L1[0],L1[1]+1) for l_ in range(L[0],L[1]+1)]), L[1]+L1[1])
    l1_min, l1_max = L1
    l2_min, l2_max = L2
    l_min, l_max = L
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    conv = Convs[i](in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda:1')

    W, X, Y, W_so2, D_in, D_out = init_test_perf(
        N, in_channels, out_channels, 
        list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
        len(ls), device='cuda:1')
    W = torch.nn.Parameter(W,requires_grad=True)
    W_so2 = torch.nn.Parameter(W_so2, requires_grad=True)
    if W.grad is not None:
        W.grad.zero_()
    if W_so2.grad is not None:
        W_so2.grad.zero_()
    test_backward(conv, W, X, Y, clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)

    ret = {}
    
    t = timeit.repeat(
        'test_backward(conv, W, X, Y,clear_basis=False, W_so2=W_so2, D_in=D_in, D_out=D_out)',
        repeat=repeat, number=number,
        globals = locals()|globals(),
        # setup='torch.cuda.empty_cache();\nconvs[i](X,Y,W,clear_basis=True)\nif W.grad is not None:\n\tW.grad.zero_()\n'
        setup='torch.cuda.empty_cache();\ntest_backward(conv, W, X, Y, clear_basis=True, W_so2=W_so2, D_in=D_in, D_out=D_out)'
    )
    print(f'conv {i+1}: {min(t)/number*1000} ms')
    ret[i+1] = min(t)/number

if __name__ == '__main__':
    N = 5
    # l_min = l1_min = l2_min = 2
    # l_max = l1_max = l2_max = 2
    l1_min, l1_max = (1,5)
    l2_min, l2_max = (0,4)
    l_min, l_max = (2,7)
    # l1_min, l1_max = (2,3)
    # l2_min, l2_max = (1,1)
    # l_min, l_max = (1,1)
    in_channels = 5
    out_channels = 2

    # check_value(N, (l1_min, l1_max), (l2_min, l2_max), (l_min, l_max), in_channels, out_channels)

    # N = 2
    # l1_min, l1_max = (0,1)
    # l2_min, l2_max = (0,1)
    # l_min, l_max = (0,1)

    # in_channels = 256
    # out_channels = 256

    # check_forward(N, (l_min, l_max), (l1_min, l1_max), (l2_min, l2_max), in_channels, out_channels)

    # N = 2
    # l1_min, l1_max = (0,3)
    # l2_min, l2_max = (0,3)
    # l_min, l_max = (0,3)

    # in_channels = 64
    # out_channels = 64

    # check_backward(N, (l_min, l_max), (l1_min, l1_max), (l2_min, l2_max), in_channels, out_channels)

    # for N in [100, 200, 500, 1000]:
    #     for channel in [16, 32, 64, 128, 256]:
    #         for L in [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10)]:
    #             check_forward(N, L, L, L, channel, channel,reverse=True)

    # for N in [100, 200, 500, 1000]:
    #     for channel in [16, 32, 64, 128, 256]:
    #         for L in [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10)]:
    #             check_backward(N, L, L, L, channel, channel,reverse=True)

    # print('reverse')


    # for N in [100, 200, 500, 1000]:
    #     for channel in [16, 32, 64, 128, 256]:
    #         for L in [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10)]:
    #             check_forward(N, L, L, L, channel, channel,reverse=False)

    # for N in [100, 200, 500, 1000]:
    #     for channel in [16, 32, 64, 128, 256]:
    #         for L in [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10)]:
    #             check_backward(N, L, L, L, channel, channel,reverse=False)

    
    # check_forward_single(4, 200, (0,2), (0,2), (0,2), 128,128)
