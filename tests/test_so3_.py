import timeit
import torch
import torch_geometric
import e3nn
import torch_scatter

from typing import Tuple, List

import torch_geometric.utils

import sys
sys.path.append('..')

from equitorch.nn._linear import SO3Linear
from equitorch.utils._clebsch_gordan import blocked_CG, coo_CG, dense_CG
from equitorch.utils._geometries import rot_on

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
    def conv(X:torch.Tensor, Y:torch.Tensor, W:torch.Tensor, clear_basis=False):
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

    
def init_test_eq(N, in_channels, out_channels, l_l, l1_l, l2_l, len_W, device='cpu'):
    X = torch.randn(N,num_order_between(l1_l[0], l1_l[-1]),in_channels)
    r = torch.randn(N,3,dtype=float_type)
    r = r / r.norm(dim=-1, keepdim=True)
    Y2 = e3nn.o3.spherical_harmonics(l2_l, r, False)
    a,b,c = e3nn.o3.rand_angles(N,dtype=float_type)
    D1 = e3nn.math.direct_sum(*(e3nn.o3.wigner_D(l1_,a,b,c) for l1_ in l1_l))
    D2 = e3nn.math.direct_sum(*(e3nn.o3.wigner_D(l2_,a,b,c) for l2_ in l2_l))
    XR = D1@X
    Y2R = (D2 @ Y2.unsqueeze(-1)).squeeze(-1)
    D = e3nn.math.direct_sum(*(e3nn.o3.wigner_D(l_,a,b,c) for l_ in l_l)).type(float_type)
    W = torch.randn(N, len_W,in_channels, out_channels, dtype=float_type)
    # W = torch.ones(N, len_W,in_channels, out_channels, dtype=float_type)
    return W.to(device), X.to(device), Y2.to(device), \
        XR.to(device), Y2R.to(device), \
        (a.to(device),b.to(device),c.to(device)), \
        D1.to(device), D2.to(device), D.to(device)

def init_test_perf(N, in_channels, out_channels, l_l, l1_l, l2_l, len_W, device='cpu'):
    r1 = torch.randn(N,in_channels,3,dtype=float_type)
    r1 = r1 / r1.norm(dim=-1, keepdim=True)
    r2 = torch.randn(N,3,dtype=float_type)
    r2 = r2 / r2.norm(dim=-1, keepdim=True)
    Y1 = e3nn.o3.spherical_harmonics(l1_l, r1, False)
    Y2 = e3nn.o3.spherical_harmonics(l2_l, r2, False)
    W = torch.randn(N, len_W,in_channels, out_channels, dtype=float_type)
    # W = torch.ones(N, len_W,in_channels, out_channels, dtype=float_type)
    return W.to(device), Y1.transpose(-1,-2).to(device), Y2.to(device)


def check_equiv(conv, W, X, Y, XR, YR, D, tol = 1e-8):
    Z = conv(X, Y, W)
    ZR = conv(XR, YR, W, clear_basis=True)
    return torch.allclose(D@Z, ZR, atol=tol)

def check_same(convs, W, X, Y, tol=1e-8):
    Zs = [conv(X,Y,W,clear_basis=True) for conv in convs]
    return [[torch.allclose(Z1, Z2, atol=tol) for Z1 in Zs] for Z2 in Zs]

def check_value(N, L1, L2, L, in_channels, out_channels):

    print('- Check value: consistency and equivariance')
    print(f'- setting: N={N}, L={L}, L1={L1}, L2={L2}, in={in_channels}, out={out_channels}')
    print('-'*100)

    l1_min, l1_max = L1
    l2_min, l2_max = L2
    l_min, l_max = L
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))

    # print('init conv1')
    conv1 = SO3Conv1(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max)
    # print('init conv2')
    conv2 = SO3Conv2(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max)
    # print('init conv3')
    conv3 = SO3Conv3(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max)
    # print('init conv4')
    conv4 = SO3Conv4(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max)
    # print('init conv5')
    conv5 = SO3Conv5(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max)
    # conv6 = SO3Conv6(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max)

    convs = [conv1, conv2, conv3, conv4, conv5]

    # print('init vars')
    W, X, Y, XR, YR, (a,b,c), D1, D2, D = init_test_eq(
        N, in_channels, out_channels, 
        list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
        len(ls))

    for i in range(5):
        print(f'equiv {i+1}: {check_equiv(convs[i], W, X, Y, XR, YR, D, tol=1e-4)}')
    # print(f'equiv {5}: {check_equiv(convs[4], W, X.transpose(-1,-2), Y, XR, YR, D, tol=1e-4)}')
    
    # print(f'same - tol=1e-11: {check_same(convs[:-1], W, X, Y, tol=1e-11)}')
    # print(f'same - tol=1e-8: {check_same(convs[:-1], W, X, Y, tol=1e-8)}')
    print(f'same - tol=1e-8: {check_same(convs, W, X, Y, tol=1e-8)}')
    print(f'same - tol=1e-7: {check_same(convs, W, X, Y, tol=1e-7)}')
    print(f'same - tol=1e-6: {check_same(convs, W, X, Y, tol=1e-6)}')
    print(f'same - tol=1e-4: {check_same(convs, W, X, Y, tol=1e-4)}')
    # print(f'same - tol=1e-8: {check_same(convs[:-1], W, X, Y, tol=1e-8)}')
    # check_equiv(conv1, W, X, Y, XR, YR, D, tol=1e-7)

    print('-'*100)

number = 1
repeat = 100

def check_forward(N, L1, L2, L, in_channels, out_channels, reverse=False):
    print('- Check forward performance')
    print(f'- setting: N={N}, L={L}, L1={L1}, L2={L2}, in={in_channels}, out={out_channels}, reverse={reverse}')
    print('-'*100)

    l1_min, l1_max = L1
    l2_min, l2_max = L2
    l_min, l_max = L
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    torch.cuda.empty_cache()
    # print('init conv1')
    conv1 = SO3Conv1(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv2')
    conv2 = SO3Conv2(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv3')
    conv3 = SO3Conv3(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv4')
    conv4 = SO3Conv4(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv5')
    conv5 = SO3Conv5(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    conv6 = SO3Conv6(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')

    convs = [conv1, conv2, conv3, conv4, conv5, conv6]

    # print('init vars')
    try:
        W, X, Y, = init_test_perf(
            N, in_channels, out_channels, 
            list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
            len(ls), device='cuda')
    except Exception as e:
        print(e.__traceback__.format_exc())
        return {}
    rg = range(len(convs)-1,-1,-1) if reverse else range(len(convs))
    for i in rg:
        # torch.cuda.empty_cache()
        try:
            # torch.cuda.empty_cache()
            convs[i](X, Y, W)
        except Exception as e:
            print(e.__traceback__.format_exc())
            continue            
        # print(f'equiv {i+1}: {check_equiv(convs[i], W, X, Y, XR, YR, D, tol=1e-4)}')
    
    ret = {}
    
    rg = range(len(convs)-1,-1,-1) if reverse else range(len(convs))
    for i in rg:
        # print(i+1)
        # torch.cuda.empty_cache()
        try:
            # torch.cuda.empty_cache()
            t = timeit.repeat(
                'convs[i](X, Y, W,clear_basis=False)', repeat=repeat, number=number,
                # globals={'convs': convs, 'X':X, 'Y':Y, 'W':W, 'i':i, 'torch':torch},
                globals = locals()|globals(),
                # setup='torch.cuda.empty_cache();\nconvs[i](X,Y,W,clear_basis=True)\n'
                setup='convs[i](X,Y,W,clear_basis=True);\n'
            )
            
            print(f'conv {i+1}: {min(t)/number*1000} ms')
            ret[i+1] = min(t)/number
        except Exception as e:
            print(f'conv {i+1}: '+str(e.__traceback__.format_exc()))
            ret[i+1] = float('nan')
            continue

    print('-'*100)

Convs = [SO3Conv1, SO3Conv2, SO3Conv3, SO3Conv4, SO3Conv5, SO3Conv6]

def check_forward_single(i, N, L1, L2, L, in_channels, out_channels):
    # print(i+1)
    l1_min, l1_max = L1
    l2_min, l2_max = L2
    l_min, l_max = L
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    conv = Convs[i](in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')

    W, X, Y, = init_test_perf(
        N, in_channels, out_channels, 
        list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
        len(ls), device='cuda')
    conv(X, Y, W)
    
    ret = {}
    
    try:
        # torch.cuda.empty_cache()
        t = timeit.repeat(
            'conv(X, Y, W,clear_basis=False)', repeat=repeat, number=number,
            # globals={'convs': convs, 'X':X, 'Y':Y, 'W':W, 'i':i, 'torch':torch},
            globals = locals()|globals(),
            # setup='torch.cuda.empty_cache();\nconvs[i](X,Y,W,clear_basis=True)\n'
            setup='torch.cuda.empty_cache();\nconv(X,Y,W,clear_basis=True);\n'
        )
        
        print(f'conv {i+1}: {min(t)/number*1000} ms')
        ret[i+1] = min(t)/number
    except Exception as e:
        print(f'conv {i+1}: '+str(e.__traceback__.format_exc()))
        ret[i+1] = float('nan')


def test_backward(conv, W, X, Y, **kwargs):
    # print(W)
    y = conv(X, Y, W, **kwargs).sum()
    y.backward()

def check_backward(N, L1, L2, L, in_channels, out_channels, reverse=False):
    torch.cuda.empty_cache()
    print('- Check backward performance')
    print(f'- setting: N={N}, L={L}, L1={L1}, L2={L2}, in={in_channels}, out={out_channels}, reverse={reverse}')
    print('-'*100)

    l1_min, l1_max = L1
    l2_min, l2_max = L2
    l_min, l_max = L

    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))

    # print('init conv1')
    conv1 = SO3Conv1(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv2')
    conv2 = SO3Conv2(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv3')
    conv3 = SO3Conv3(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv4')
    conv4 = SO3Conv4(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    # print('init conv5')
    conv5 = SO3Conv5(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')
    conv6 = SO3Conv6(in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')

    convs = [conv1, conv2, conv3, conv4, conv5, conv6]

    # print('init vars')
    try:
        W, X, Y, = init_test_perf(
            N, in_channels, out_channels, 
            list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
            len(ls), device='cuda')
        W = torch.nn.Parameter(W,requires_grad=True)
    except Exception as e:
        print(e.__traceback__.format_exc())
        return {}
    rg = range(len(convs)-1,-1,-1) if reverse else range(len(convs))
    for i in rg:
        try:
            # torch.cuda.empty_cache()
            if W.grad is not None:
                W.grad.zero_()
            convs[i](X, Y, W)
        except Exception as e:
            print(f'conv {i+1}: '+str(e.__traceback__.format_exc()))
            continue            
        # print(f'equiv {i+1}: {check_equiv(convs[i], W, X, Y, XR, YR, D, tol=1e-4)}')

    rg = range(len(convs)-1,-1,-1) if reverse else range(len(convs))

    ret = {}

    for i in rg:
        # print(i+1)
        # torch.cuda.empty_cache()
        try:
            # torch.cuda.empty_cache()
            t = timeit.repeat(
                'test_backward(convs[i], W, X, Y,clear_basis=False)',
                repeat=repeat, number=number,
                globals = locals()|globals(),
                # setup='torch.cuda.empty_cache();\nconvs[i](X,Y,W,clear_basis=True)\nif W.grad is not None:\n\tW.grad.zero_()\n'
                setup='convs[i](X,Y,W,clear_basis=True)'
            )
            # globals
            print(f'conv {i+1}: {min(t)/number*1000} ms')
            ret[i+1] = min(t)/number
        except Exception as e:
            print(e.__traceback__.format_exc())
            ret[i+1] = float('nan')
            continue
    print('-'*100)

def check_backward_single(i, N, L1, L2, L, in_channels, out_channels):
    # print(i+1)
    l1_min, l1_max = L1
    l2_min, l2_max = L2
    l_min, l_max = L
    ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
    conv = Convs[i](in_channels, out_channels, l_min, l_max, l1_min, l1_max, l2_min, l2_max, device='cuda')

    W, X, Y, = init_test_perf(
        N, in_channels, out_channels, 
        list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
        len(ls), device='cuda')
    W = torch.nn.Parameter(W,requires_grad=True)
    if W.grad is not None:
        W.grad.zero_()
    test_backward(conv, W, X, Y)

    ret = {}
    
    t = timeit.repeat(
        'test_backward(conv, W, X, Y,clear_basis=False)',
        repeat=repeat, number=number,
        globals = locals()|globals(),
        # setup='torch.cuda.empty_cache();\nconvs[i](X,Y,W,clear_basis=True)\nif W.grad is not None:\n\tW.grad.zero_()\n'
        setup='torch.cuda.empty_cache();\ntest_backward(conv, W, X, Y, clear_basis=True)'
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
