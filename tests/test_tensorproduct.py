import torch
from torch import Tensor

from typing import List, Union

from e3nn.o3 import FullyConnectedTensorProduct 
from e3nn.o3 import wigner_3j

from ..equitorch.nn._tensor_product import _tensor_product


@torch.jit.script
def e3nn_tp(tp: FullyConnectedTensorProduct, H1: Tensor, H2: Tensor, repeat:int) -> List[Tensor]:
    results = []
    for i in range(repeat):
        results.append(tp(H1.transpose(-1,-2), H2.transpose(-1,-2)))
    return results

@torch.jit.script
def my_tp(w: Tensor, cg: Tensor, H1: Tensor, H2: Tensor, repeat:int) -> List[Tensor]:
    res = []
    for i in range(repeat):
        res.append()
    return res

def check_same(results, result, L):
    same = True
    for l in range(L+1):
        same = same and results[l].allclose(result[:,l**2:(l+1)**2,l**2:(l+1)**2])
    return same

warmup_times = 10
repeat_times = 10
repeat = 1

device = 'cuda:1'

def test(N,L):
    times_block = []
    times_full = []
    for t in range(warmup_times):
        R = o3.rand_matrix(N)
        Ds, D = process_Ds(L, R, device=device)
        vec = torch.randn(N,(L+1)**2,1).to(device)
        vecs = [vec[:,l**2:(l+1)**2,:].contiguous() for l in range(L+1)]
        block_wise_mult(Ds, vecs, repeat=repeat)
        full_mult(D, vec, repeat=repeat)
    for t in range(repeat_times):
        R = o3.rand_matrix(N)
        Ds, D = process_Ds(L, R, device=device)
        vec = torch.randn(N,(L+1)**2,1).to(device)
        vecs = [vec[:,l**2:(l+1)**2,:].contiguous() for l in range(L+1)]
        
        time1 = time.time()
        results = block_wise_mult(Ds, vecs, repeat=repeat)
        time2 = time.time()
        times_block.append(time2-time1)

        time1 = time.time()
        result = full_mult(D, vec, repeat=repeat)
        time2 = time.time()
        times_full.append(time2-time1)
        # print('same: ', check_same(results, result[0], L))
        for t in results:
            del t
        for t in result:
            del t
        torch.cuda.empty_cache()

    return times_block, times_full

res = {}

for N in [1000, 10000, 100000]:
    for L in [0, 1, 4, 8, 12, 16, 20]:
        if N == 100000 and L > 8:
            break
        print(N,L)
        times_block, times_full = test(N,L)
        print(min(times_block), sum(times_block) / repeat_times)
        print(min(times_full), sum(times_full) / repeat_times)
        res[(N,L)] = {'min_block': min(times_block), 'mean_block':sum(times_block) / repeat_times, 'min_full':min(times_full), 'mean_full':sum(times_full) / repeat_times}
        print()
        

torch.save(res, './test_D_mult_block_res.pkl')