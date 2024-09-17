# # %
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('..')




import torch
from torch import Tensor

import torch_geometric
import e3nn

from equitorch.nn._linear import SO2Linear, SO3Linear
from equitorch.typing import DegreeRange
from equitorch.utils._clebsch_gordan import blocked_CG
from equitorch.utils._geometries import rot_on
from equitorch.utils._indices import list_degrees
from equitorch.math._o3 import spherical_harmonics, wigner_D
from equitorch.utils._geometries import align_to_z_wigner
from equitorch.utils._other import so3_weights_to_so2, so2_weights_to_so3

float_type = torch.float64

# torch.random.manual_seed(0)

# %
def init_test(N, in_channels, out_channels, l_l, l1_l, l2_l, len_W, device='cpu'):
    r1 = torch.randn(N,in_channels,3,dtype=float_type)
    r2 = torch.randn(N,3,dtype=float_type)
    r2 = r2 / r2.norm(dim=-1, keepdim=True)
    Y1 = spherical_harmonics(r1, (l1_l[0], l1_l[-1]))
    Y2 = spherical_harmonics(r2, (l2_l[0], l2_l[-1]))
    a,b,c = e3nn.o3.rand_angles(N,dtype=float_type)
    D1 = wigner_D((l1_l[0], l1_l[-1]), a, b, c)
    D2 = wigner_D((l2_l[0], l2_l[-1]), a, b, c)
    Y1R = D1 @ Y1.transpose(-1,-2)
    Y2R = (D2 @ Y2.unsqueeze(-1)).squeeze(-1)
    D = wigner_D((l_l[0], l_l[-1]), a, b, c)
    W = torch.randn(N, len_W,in_channels, out_channels, dtype=float_type)
    # W = torch.ones(N, len_W,in_channels, out_channels, dtype=float_type)
    return W.to(device), Y1.transpose(-1,-2).to(device), Y2.to(device), \
        Y1R.to(device), Y2R.to(device), \
        (a.to(device),b.to(device),c.to(device)), \
        D1.to(device), D2.to(device), D.to(device), r2.to(device)

# %
N = 6

l1_min, l1_max = (1,4)
l_min, l_max = (2,7)
l2_min = min([abs(l1-l) for l1 in range(l1_min,l1_max+1) for l in range(l_min,l_max+1)])
l2_max = l_max+l1_max
ls = list_degrees((l_min, l_max), (l1_min, l1_max), (l2_min, l2_max))
in_channels = 2
out_channels = 5

so2 = SO2Linear(in_channels, out_channels, (l1_min, l1_max), (l_min, l_max), True, False)
so3 = SO3Linear(in_channels, out_channels, (l1_min, l1_max), (l2_min, l2_max), (l_min, l_max), True, False)


W_so3, X, Y, XR, YR, (a,b,c), D1, D2, D, r = init_test(
    N, in_channels, out_channels, 
    list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
    len(ls))

W_so2 = so3_weights_to_so2(W_so3, (l1_min, l1_max), (l_min, l_max), channel_wise=False)

D_so2_in = align_to_z_wigner(r, (l1_min, l1_max))
D_align_Y = align_to_z_wigner(r, (l2_min, l2_max))
D_so2_out = align_to_z_wigner(r, (l_min, l_max))

Z_so2 = rot_on(D_so2_out.transpose(-1,-2), so2.forward(rot_on(D_so2_in, X), W_so2))
Z_so3 = so3.forward(X, Y, W_so3)

print(((Z_so2 / Z_so3)-1).abs().mean())






W_so2, X, Y, XR, YR, (a,b,c), D1, D2, D, r = init_test(
    N, in_channels, out_channels, 
    list(range(l_min, l_max+1)), list(range(l1_min, l1_max+1)), list(range(l2_min, l2_max+1)),
    len(ls))

W_so3 = so2_weights_to_so3(W_so2, (l1_min, l1_max), (l_min, l_max), channel_wise=False)

D_so2_in = align_to_z_wigner(r, (l1_min, l1_max))
D_align_Y = align_to_z_wigner(r, (l2_min, l2_max))
D_so2_out = align_to_z_wigner(r, (l_min, l_max))

Z_so2 = rot_on(D_so2_out.transpose(-1,-2), so2.forward(rot_on(D_so2_in, X), W_so2))
Z_so3 = so3.forward(X, Y, W_so3)

print(((Z_so2 / Z_so3)-1).abs().mean())
