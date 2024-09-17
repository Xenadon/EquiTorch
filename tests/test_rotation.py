import torch

import sys
sys.path.append('..')

from equitorch.math import spherical_harmonics
from equitorch.utils._geometries import align_to_z_mat, align_to_z_wigner, rot_on

L = (0,5)

r = torch.eye(3, 3)
# r = r / r.norm(dim=-1,keepdim=True)

R = align_to_z_mat(r, False)

print(rot_on(R,r))

Y = spherical_harmonics(r,L)
D = align_to_z_wigner(r, L, False)
DY = rot_on(D, Y)
print(DY)
print(DY.norm(dim=-1 ))
print((DY>0.1).type(torch.int))

print('-'*50)

L = (2,5)

r = torch.randn(5, 3)
r = r / r.norm(dim=-1,keepdim=True)

R = align_to_z_mat(r, False)

print(rot_on(R,r))

Y = spherical_harmonics(r,L)
D = align_to_z_wigner(r, L, False)
DY = rot_on(D, Y)
print(DY)
print(DY.norm(dim=-1 ))
print((DY>0.1).type(torch.int))