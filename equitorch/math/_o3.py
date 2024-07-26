from typing import Union, List, Any

import math

import sympy
from sympy.printing.pycode import pycode
import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from ..utils._indices import check_degree_range, num_order_between

from ..typing import DegreeRange

# all the functions below are borrowed and modified form https://github.com/e3nn/e3nn/tree/0.5.0/e3nn/o3

def angles_to_xyz(theta, phi, dim=-1):
    r"""Convert spherical coordinates :math:`(\theta, \phi)` into Cartesian coordinates :math:`(x, y, z)` on the unit sphere.

    Uses the physics convention of spherical coordinates.
    See: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Parameters
    ----------
    theta : torch.Tensor
        The polar angles, tensor of shape :math:`(...)`
    phi : torch.Tensor
        The azimuthal angles, tensor of shape :math:`(...)`
    dim : int, optional
        The dimension along which to stack the result. Defaults to -1.

    Returns
    -------
    torch.Tensor
        Cartesian coordinates, tensor of shape :math:`(..., 3)`
    """
    alpha, beta = torch.broadcast_tensors(alpha, beta)
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=dim)


def xyz_to_angles(xyz, need_normalize, dim=-1):
    r"""Convert Cartesian coordinates :math:`(x, y, z)` to spherical coordinates :math:`(\theta, \phi)`.

    Follows the equation:

    .. math::

        \vec{r} = R_z(\phi) R_y(\theta) \vec{e}_z

    ::
    where :math:`R_z` and :math:`R_y` are rotation matrices, and :math:`\vec{e}_z` is the unit vector along z-axis.

    Parameters
    ----------
    xyz : torch.Tensor
        Cartesian coordinates, tensor of shape :math:`(..., 3)`
    need_normalize : bool
        Whether to normalize the input coordinates
    dim : int, optional
        The dimension containing the xyz coordinates. Defaults to -1.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - phi : Azimuthal angle, tensor of shape :math:`(...)`
        - theta : Polar angle, tensor of shape :math:`(...)`
    """
    if need_normalize:
        xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
        xyz = xyz.clamp(-1.00001, 1.00001)

    theta = torch.acos(xyz.select(dim, 2))
    phi = torch.atan2(xyz.select(dim,1), xyz.select(dim,0))
    return phi, theta


# borrowed and modified form e3nn
def angles_to_matrix(alpha, beta, gamma):
    r"""Convert Z-Y-Z Euler angles to rotation matrices.

    Parameters
    ----------
    alpha : torch.Tensor
        First rotation angle around Z-axis, tensor of shape :math:`(...)`
    beta : torch.Tensor
        Second rotation angle around Y-axis, tensor of shape :math:`(...)`
    gamma : torch.Tensor
        Third rotation angle around Z-axis, tensor of shape :math:`(...)`

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return o3.matrix_z(alpha) @ o3.matrix_y(beta) @ o3.matrix_z(gamma)

def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)` parameterized by Z-Y-Z angles.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Z axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around Y axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Z axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = o3._wigner.so3_generators(l)
    # return torch.matrix_exp(alpha * X[2]) @ torch.matrix_exp(beta * X[1]) @ torch.matrix_exp(gamma * X[2])
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])


def spherical_harmonics(X: torch.Tensor, 
                        L: DegreeRange, 
                        need_normalize:bool=False,
                        dim:int=-1):
    r"""Compute spherical harmonics for given Cartesian coordinates.

    Gives the spherical harmonics in the degree range L of exactly the same form 
    as in https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Sphericalfunctions.svg,
    where :math:`theta` and :math:`phi` are the polar and azimuth angles respectively.
    The harmonics are ordered as m = -l, -l+1, ..., l-1, l for each degree l. 

    Examples:
        * l = 0 ~ [1]
        * l = 1 ~ [y, z, x]
        * l = 2 ~ [xy, yz, z^2-x^2-y^2, xz, x^2-y^2] (with different coefficients)

    If there is a channel dimension in X, it will also be included in the returned spherical harmonics.

    Parameters
    ----------
    X : torch.Tensor
        The coordinate tensor.
    L : DegreeRange
        The degree range of spherical harmonics that we need.
        The maximum degree in L should not execeed 13.
    need_normalize : bool, optional
        Whether the input needs to be normalized. Default is False.
    dim : int, optional
        The 'xyz' dimension in X. Default is -1.

    Returns
    -------
    torch.Tensor
        Tensor of spherical harmonics.

    Notes
    -----
    The shape of the output tensor depends on the input shape and the degree range L.
    """

    if need_normalize:
        X = X / X.norm(p='fro',dim=dim,keepdim=True)
    lmin, lmax = check_degree_range(L)
    return _spherical_harmonics(lmax, X,dim).narrow(dim, lmin**2, num_order_between(lmin, lmax))

def _generate_spherical_harmonics(lmax, device=None):  # pragma: no cover
    r"""code used to generate the code above

    based on `wigner_3j`
    """
    torch.set_default_dtype(torch.float64)

    print("""x = X.narrow(dim,0,1)\ny = X.narrow(dim,1,1)\nz = X.narrow(dim,2,1)\n""")

    def to_frac(x: float):
        from fractions import Fraction

        s = 1 if x >= 0 else -1
        x = x**2
        x = Fraction(x).limit_denominator()
        x = s * sympy.sqrt(x)
        x = sympy.simplify(x)
        return x

    print("sh_0_0 = torch.ones_like(x)")
    print("if lmax == 0:")
    print("    return torch.cat([")
    print("        sh_0_0,")
    print("    ], dim=dim)")
    print()

    x_var, y_var, z_var = sympy.symbols("y z x")
    polynomials = [sympy.sqrt(3) * x_var, sympy.sqrt(3) * y_var, sympy.sqrt(3) * z_var]
    
    def sub_z1(p, names, polynormz):
        p = p.subs(x_var, 0).subs(y_var, 1).subs(z_var, 0)
        for n, c in zip(names, polynormz):
            p = p.subs(n, c)
        return p

    poly_evalz = [sub_z1(p, [], []) for p in polynomials]

    for l in range(1, lmax + 1):
        sh_variables = sympy.symbols(" ".join(f"sh_{l}_{m}" for m in range(2 * l + 1)))

        for n, p in zip(sh_variables, polynomials):
            print(f"{n} = {pycode(p)}")

        print(f"if lmax == {l}:")
        u = ",\n        ".join(", ".join(f"sh_{j}_{m}" for m in range(2 * j + 1)) for j in range(l + 1))
        print(f"    return torch.cat([\n        {u}\n    ], dim=dim)")
        print()

        if l == lmax:
            break

        polynomials = [
            sum(to_frac(c.item()) * v * sh for cj, v in zip(cij, [x_var, y_var, z_var]) for c, sh in zip(cj, sh_variables))
            for cij in o3.wigner_3j(l + 1, 1, l, device=device)
        ]

        poly_evalz = [sub_z1(p, sh_variables, poly_evalz) for p in polynomials]
        norm = sympy.sqrt(sum(p**2 for p in poly_evalz))
        polynomials = [sympy.sqrt(2 * l + 3) * p / norm for p in polynomials]
        poly_evalz = [sympy.sqrt(2 * l + 3) * p / norm for p in poly_evalz]

        polynomials = [sympy.simplify(p, full=True) for p in polynomials]

@torch.jit.script
def _spherical_harmonics(lmax:int, X:torch.Tensor,dim:int=-2):
    x = X.narrow(dim,0,1)
    y = X.narrow(dim,1,1)
    z = X.narrow(dim,2,1)

    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.cat([
            sh_0_0,
        ], dim=dim)

    sh_1_0 = math.sqrt(3)*y
    sh_1_1 = math.sqrt(3)*z
    sh_1_2 = math.sqrt(3)*x
    if lmax == 1:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2
        ], dim=dim)

    sh_2_0 = (1/2)*math.sqrt(5)*(sh_1_0*x + sh_1_2*y)
    sh_2_1 = (1/2)*math.sqrt(5)*(sh_1_0*z + sh_1_1*y)
    sh_2_2 = (1/6)*math.sqrt(15)*(-sh_1_0*y + 2*sh_1_1*z - sh_1_2*x)
    sh_2_3 = (1/2)*math.sqrt(5)*(sh_1_1*x + sh_1_2*z)
    sh_2_4 = (1/2)*math.sqrt(5)*(-sh_1_0*y + sh_1_2*x)
    if lmax == 2:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=dim)

    sh_3_0 = (1/6)*math.sqrt(42)*(sh_2_0*x + sh_2_4*y)
    sh_3_1 = (1/3)*math.sqrt(7)*(sh_2_0*z + sh_2_1*x + sh_2_3*y)
    sh_3_2 = -1/30*math.sqrt(70)*sh_2_0*x + (2/15)*math.sqrt(70)*sh_2_1*z + (1/15)*math.sqrt(210)*sh_2_2*y + (1/30)*math.sqrt(70)*sh_2_4*y
    sh_3_3 = -1/15*math.sqrt(105)*sh_2_1*y + (1/5)*math.sqrt(35)*sh_2_2*z - 1/15*math.sqrt(105)*sh_2_3*x
    sh_3_4 = -1/30*math.sqrt(70)*sh_2_0*y + (1/15)*math.sqrt(210)*sh_2_2*x + (2/15)*math.sqrt(70)*sh_2_3*z - 1/30*math.sqrt(70)*sh_2_4*x
    sh_3_5 = (1/3)*math.sqrt(7)*(-sh_2_1*y + sh_2_3*x + sh_2_4*z)
    sh_3_6 = (1/6)*math.sqrt(42)*(-sh_2_0*y + sh_2_4*x)
    if lmax == 3:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
        ], dim=dim)

    sh_4_0 = (3/4)*math.sqrt(2)*(sh_3_0*x + sh_3_6*y)
    sh_4_1 = (3/4)*sh_3_0*z + (3/8)*math.sqrt(6)*sh_3_1*x + (3/8)*math.sqrt(6)*sh_3_5*y
    sh_4_2 = -3/56*math.sqrt(14)*sh_3_0*x + (3/14)*math.sqrt(21)*sh_3_1*z + (3/56)*math.sqrt(210)*sh_3_2*x + (3/56)*math.sqrt(210)*sh_3_4*y + (3/56)*math.sqrt(14)*sh_3_6*y
    sh_4_3 = -3/56*math.sqrt(42)*sh_3_1*x + (3/28)*math.sqrt(105)*sh_3_2*z + (3/28)*math.sqrt(70)*sh_3_3*y + (3/56)*math.sqrt(42)*sh_3_5*y
    sh_4_4 = -3/28*math.sqrt(42)*sh_3_2*y + (3/7)*math.sqrt(7)*sh_3_3*z - 3/28*math.sqrt(42)*sh_3_4*x
    sh_4_5 = -3/56*math.sqrt(42)*sh_3_1*y + (3/28)*math.sqrt(70)*sh_3_3*x + (3/28)*math.sqrt(105)*sh_3_4*z - 3/56*math.sqrt(42)*sh_3_5*x
    sh_4_6 = -3/56*math.sqrt(14)*sh_3_0*y - 3/56*math.sqrt(210)*sh_3_2*y + (3/56)*math.sqrt(210)*sh_3_4*x + (3/14)*math.sqrt(21)*sh_3_5*z - 3/56*math.sqrt(14)*sh_3_6*x
    sh_4_7 = -3/8*math.sqrt(6)*sh_3_1*y + (3/8)*math.sqrt(6)*sh_3_5*x + (3/4)*sh_3_6*z
    sh_4_8 = (3/4)*math.sqrt(2)*(-sh_3_0*y + sh_3_6*x)
    if lmax == 4:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
        ], dim=dim)

    sh_5_0 = (1/10)*math.sqrt(110)*(sh_4_0*x + sh_4_8*y)
    sh_5_1 = (1/5)*math.sqrt(11)*sh_4_0*z + (1/5)*math.sqrt(22)*sh_4_1*x + (1/5)*math.sqrt(22)*sh_4_7*y
    sh_5_2 = -1/30*math.sqrt(22)*sh_4_0*x + (4/15)*math.sqrt(11)*sh_4_1*z + (1/15)*math.sqrt(154)*sh_4_2*x + (1/15)*math.sqrt(154)*sh_4_6*y + (1/30)*math.sqrt(22)*sh_4_8*y
    sh_5_3 = -1/30*math.sqrt(66)*sh_4_1*x + (1/15)*math.sqrt(231)*sh_4_2*z + (1/30)*math.sqrt(462)*sh_4_3*x + (1/30)*math.sqrt(462)*sh_4_5*y + (1/30)*math.sqrt(66)*sh_4_7*y
    sh_5_4 = -1/15*math.sqrt(33)*sh_4_2*x + (2/15)*math.sqrt(66)*sh_4_3*z + (1/15)*math.sqrt(165)*sh_4_4*y + (1/15)*math.sqrt(33)*sh_4_6*y
    sh_5_5 = -1/15*math.sqrt(110)*sh_4_3*y + (1/3)*math.sqrt(11)*sh_4_4*z - 1/15*math.sqrt(110)*sh_4_5*x
    sh_5_6 = -1/15*math.sqrt(33)*sh_4_2*y + (1/15)*math.sqrt(165)*sh_4_4*x + (2/15)*math.sqrt(66)*sh_4_5*z - 1/15*math.sqrt(33)*sh_4_6*x
    sh_5_7 = -1/30*math.sqrt(66)*sh_4_1*y - 1/30*math.sqrt(462)*sh_4_3*y + (1/30)*math.sqrt(462)*sh_4_5*x + (1/15)*math.sqrt(231)*sh_4_6*z - 1/30*math.sqrt(66)*sh_4_7*x
    sh_5_8 = -1/30*math.sqrt(22)*sh_4_0*y - 1/15*math.sqrt(154)*sh_4_2*y + (1/15)*math.sqrt(154)*sh_4_6*x + (4/15)*math.sqrt(11)*sh_4_7*z - 1/30*math.sqrt(22)*sh_4_8*x
    sh_5_9 = -1/5*math.sqrt(22)*sh_4_1*y + (1/5)*math.sqrt(22)*sh_4_7*x + (1/5)*math.sqrt(11)*sh_4_8*z
    sh_5_10 = (1/10)*math.sqrt(110)*(-sh_4_0*y + sh_4_8*x)
    if lmax == 5:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
        ], dim=dim)

    sh_6_0 = (1/6)*math.sqrt(39)*(sh_5_0*x + sh_5_10*y)
    sh_6_1 = (1/6)*math.sqrt(13)*sh_5_0*z + (1/12)*math.sqrt(130)*sh_5_1*x + (1/12)*math.sqrt(130)*sh_5_9*y
    sh_6_2 = -1/132*math.sqrt(286)*sh_5_0*x + (1/33)*math.sqrt(715)*sh_5_1*z + (1/132)*math.sqrt(286)*sh_5_10*y + (1/44)*math.sqrt(1430)*sh_5_2*x + (1/44)*math.sqrt(1430)*sh_5_8*y
    sh_6_3 = -1/132*math.sqrt(858)*sh_5_1*x + (1/22)*math.sqrt(429)*sh_5_2*z + (1/22)*math.sqrt(286)*sh_5_3*x + (1/22)*math.sqrt(286)*sh_5_7*y + (1/132)*math.sqrt(858)*sh_5_9*y
    sh_6_4 = -1/66*math.sqrt(429)*sh_5_2*x + (2/33)*math.sqrt(286)*sh_5_3*z + (1/66)*math.sqrt(2002)*sh_5_4*x + (1/66)*math.sqrt(2002)*sh_5_6*y + (1/66)*math.sqrt(429)*sh_5_8*y
    sh_6_5 = -1/66*math.sqrt(715)*sh_5_3*x + (1/66)*math.sqrt(5005)*sh_5_4*z + (1/66)*math.sqrt(3003)*sh_5_5*y + (1/66)*math.sqrt(715)*sh_5_7*y
    sh_6_6 = -1/66*math.sqrt(2145)*sh_5_4*y + (1/11)*math.sqrt(143)*sh_5_5*z - 1/66*math.sqrt(2145)*sh_5_6*x
    sh_6_7 = -1/66*math.sqrt(715)*sh_5_3*y + (1/66)*math.sqrt(3003)*sh_5_5*x + (1/66)*math.sqrt(5005)*sh_5_6*z - 1/66*math.sqrt(715)*sh_5_7*x
    sh_6_8 = -1/66*math.sqrt(429)*sh_5_2*y - 1/66*math.sqrt(2002)*sh_5_4*y + (1/66)*math.sqrt(2002)*sh_5_6*x + (2/33)*math.sqrt(286)*sh_5_7*z - 1/66*math.sqrt(429)*sh_5_8*x
    sh_6_9 = -1/132*math.sqrt(858)*sh_5_1*y - 1/22*math.sqrt(286)*sh_5_3*y + (1/22)*math.sqrt(286)*sh_5_7*x + (1/22)*math.sqrt(429)*sh_5_8*z - 1/132*math.sqrt(858)*sh_5_9*x
    sh_6_10 = -1/132*math.sqrt(286)*sh_5_0*y - 1/132*math.sqrt(286)*sh_5_10*x - 1/44*math.sqrt(1430)*sh_5_2*y + (1/44)*math.sqrt(1430)*sh_5_8*x + (1/33)*math.sqrt(715)*sh_5_9*z
    sh_6_11 = -1/12*math.sqrt(130)*sh_5_1*y + (1/6)*math.sqrt(13)*sh_5_10*z + (1/12)*math.sqrt(130)*sh_5_9*x
    sh_6_12 = (1/6)*math.sqrt(39)*(-sh_5_0*y + sh_5_10*x)
    if lmax == 6:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
        ], dim=dim)

    sh_7_0 = (1/14)*math.sqrt(210)*(sh_6_0*x + sh_6_12*y)
    sh_7_1 = (1/7)*math.sqrt(15)*sh_6_0*z + (3/7)*math.sqrt(5)*sh_6_1*x + (3/7)*math.sqrt(5)*sh_6_11*y
    sh_7_2 = -1/182*math.sqrt(390)*sh_6_0*x + (6/91)*math.sqrt(130)*sh_6_1*z + (3/91)*math.sqrt(715)*sh_6_10*y + (1/182)*math.sqrt(390)*sh_6_12*y + (3/91)*math.sqrt(715)*sh_6_2*x
    sh_7_3 = -3/182*math.sqrt(130)*sh_6_1*x + (3/182)*math.sqrt(130)*sh_6_11*y + (3/91)*math.sqrt(715)*sh_6_2*z + (5/182)*math.sqrt(858)*sh_6_3*x + (5/182)*math.sqrt(858)*sh_6_9*y
    sh_7_4 = (3/91)*math.sqrt(65)*sh_6_10*y - 3/91*math.sqrt(65)*sh_6_2*x + (10/91)*math.sqrt(78)*sh_6_3*z + (15/182)*math.sqrt(78)*sh_6_4*x + (15/182)*math.sqrt(78)*sh_6_8*y
    sh_7_5 = -5/91*math.sqrt(39)*sh_6_3*x + (15/91)*math.sqrt(39)*sh_6_4*z + (3/91)*math.sqrt(390)*sh_6_5*x + (3/91)*math.sqrt(390)*sh_6_7*y + (5/91)*math.sqrt(39)*sh_6_9*y
    sh_7_6 = -15/182*math.sqrt(26)*sh_6_4*x + (12/91)*math.sqrt(65)*sh_6_5*z + (2/91)*math.sqrt(1365)*sh_6_6*y + (15/182)*math.sqrt(26)*sh_6_8*y
    sh_7_7 = -3/91*math.sqrt(455)*sh_6_5*y + (1/13)*math.sqrt(195)*sh_6_6*z - 3/91*math.sqrt(455)*sh_6_7*x
    sh_7_8 = -15/182*math.sqrt(26)*sh_6_4*y + (2/91)*math.sqrt(1365)*sh_6_6*x + (12/91)*math.sqrt(65)*sh_6_7*z - 15/182*math.sqrt(26)*sh_6_8*x
    sh_7_9 = -5/91*math.sqrt(39)*sh_6_3*y - 3/91*math.sqrt(390)*sh_6_5*y + (3/91)*math.sqrt(390)*sh_6_7*x + (15/91)*math.sqrt(39)*sh_6_8*z - 5/91*math.sqrt(39)*sh_6_9*x
    sh_7_10 = -3/91*math.sqrt(65)*sh_6_10*x - 3/91*math.sqrt(65)*sh_6_2*y - 15/182*math.sqrt(78)*sh_6_4*y + (15/182)*math.sqrt(78)*sh_6_8*x + (10/91)*math.sqrt(78)*sh_6_9*z
    sh_7_11 = -3/182*math.sqrt(130)*sh_6_1*y + (3/91)*math.sqrt(715)*sh_6_10*z - 3/182*math.sqrt(130)*sh_6_11*x - 5/182*math.sqrt(858)*sh_6_3*y + (5/182)*math.sqrt(858)*sh_6_9*x
    sh_7_12 = -1/182*math.sqrt(390)*sh_6_0*y + (3/91)*math.sqrt(715)*sh_6_10*x + (6/91)*math.sqrt(130)*sh_6_11*z - 1/182*math.sqrt(390)*sh_6_12*x - 3/91*math.sqrt(715)*sh_6_2*y
    sh_7_13 = -3/7*math.sqrt(5)*sh_6_1*y + (3/7)*math.sqrt(5)*sh_6_11*x + (1/7)*math.sqrt(15)*sh_6_12*z
    sh_7_14 = (1/14)*math.sqrt(210)*(-sh_6_0*y + sh_6_12*x)
    if lmax == 7:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14
        ], dim=dim)

    sh_8_0 = (1/4)*math.sqrt(17)*(sh_7_0*x + sh_7_14*y)
    sh_8_1 = (1/8)*math.sqrt(17)*sh_7_0*z + (1/16)*math.sqrt(238)*sh_7_1*x + (1/16)*math.sqrt(238)*sh_7_13*y
    sh_8_2 = -1/240*math.sqrt(510)*sh_7_0*x + (1/60)*math.sqrt(1785)*sh_7_1*z + (1/240)*math.sqrt(46410)*sh_7_12*y + (1/240)*math.sqrt(510)*sh_7_14*y + (1/240)*math.sqrt(46410)*sh_7_2*x
    sh_8_3 = (1/80)*math.sqrt(2)*(-math.sqrt(85)*sh_7_1*x + math.sqrt(2210)*sh_7_11*y + math.sqrt(85)*sh_7_13*y + math.sqrt(2210)*sh_7_2*z + math.sqrt(2210)*sh_7_3*x)
    sh_8_4 = (1/40)*math.sqrt(935)*sh_7_10*y + (1/40)*math.sqrt(85)*sh_7_12*y - 1/40*math.sqrt(85)*sh_7_2*x + (1/10)*math.sqrt(85)*sh_7_3*z + (1/40)*math.sqrt(935)*sh_7_4*x
    sh_8_5 = (1/48)*math.sqrt(2)*(math.sqrt(102)*sh_7_11*y - math.sqrt(102)*sh_7_3*x + math.sqrt(1122)*sh_7_4*z + math.sqrt(561)*sh_7_5*x + math.sqrt(561)*sh_7_9*y)
    sh_8_6 = (1/16)*math.sqrt(34)*sh_7_10*y - 1/16*math.sqrt(34)*sh_7_4*x + (1/4)*math.sqrt(17)*sh_7_5*z + (1/16)*math.sqrt(102)*sh_7_6*x + (1/16)*math.sqrt(102)*sh_7_8*y
    sh_8_7 = -1/80*math.sqrt(1190)*sh_7_5*x + (1/40)*math.sqrt(1785)*sh_7_6*z + (1/20)*math.sqrt(255)*sh_7_7*y + (1/80)*math.sqrt(1190)*sh_7_9*y
    sh_8_8 = -1/60*math.sqrt(1785)*sh_7_6*y + (1/15)*math.sqrt(255)*sh_7_7*z - 1/60*math.sqrt(1785)*sh_7_8*x
    sh_8_9 = -1/80*math.sqrt(1190)*sh_7_5*y + (1/20)*math.sqrt(255)*sh_7_7*x + (1/40)*math.sqrt(1785)*sh_7_8*z - 1/80*math.sqrt(1190)*sh_7_9*x
    sh_8_10 = -1/16*math.sqrt(34)*sh_7_10*x - 1/16*math.sqrt(34)*sh_7_4*y - 1/16*math.sqrt(102)*sh_7_6*y + (1/16)*math.sqrt(102)*sh_7_8*x + (1/4)*math.sqrt(17)*sh_7_9*z
    sh_8_11 = (1/48)*math.sqrt(2)*(math.sqrt(1122)*sh_7_10*z - math.sqrt(102)*sh_7_11*x - math.sqrt(102)*sh_7_3*y - math.sqrt(561)*sh_7_5*y + math.sqrt(561)*sh_7_9*x)
    sh_8_12 = (1/40)*math.sqrt(935)*sh_7_10*x + (1/10)*math.sqrt(85)*sh_7_11*z - 1/40*math.sqrt(85)*sh_7_12*x - 1/40*math.sqrt(85)*sh_7_2*y - 1/40*math.sqrt(935)*sh_7_4*y
    sh_8_13 = (1/80)*math.sqrt(2)*(-math.sqrt(85)*sh_7_1*y + math.sqrt(2210)*sh_7_11*x + math.sqrt(2210)*sh_7_12*z - math.sqrt(85)*sh_7_13*x - math.sqrt(2210)*sh_7_3*y)
    sh_8_14 = -1/240*math.sqrt(510)*sh_7_0*y + (1/240)*math.sqrt(46410)*sh_7_12*x + (1/60)*math.sqrt(1785)*sh_7_13*z - 1/240*math.sqrt(510)*sh_7_14*x - 1/240*math.sqrt(46410)*sh_7_2*y
    sh_8_15 = -1/16*math.sqrt(238)*sh_7_1*y + (1/16)*math.sqrt(238)*sh_7_13*x + (1/8)*math.sqrt(17)*sh_7_14*z
    sh_8_16 = (1/4)*math.sqrt(17)*(-sh_7_0*y + sh_7_14*x)
    if lmax == 8:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16
        ], dim=dim)

    sh_9_0 = (1/6)*math.sqrt(38)*(sh_8_0*x + sh_8_16*y)
    sh_9_1 = (1/9)*math.sqrt(19)*(sh_8_0*z + 2*sh_8_1*x + 2*sh_8_15*y)
    sh_9_2 = -1/306*math.sqrt(646)*sh_8_0*x + (4/153)*math.sqrt(646)*sh_8_1*z + (2/153)*math.sqrt(4845)*sh_8_14*y + (1/306)*math.sqrt(646)*sh_8_16*y + (2/153)*math.sqrt(4845)*sh_8_2*x
    sh_9_3 = -1/306*math.sqrt(1938)*sh_8_1*x + (1/306)*math.sqrt(67830)*sh_8_13*y + (1/306)*math.sqrt(1938)*sh_8_15*y + (1/51)*math.sqrt(1615)*sh_8_2*z + (1/306)*math.sqrt(67830)*sh_8_3*x
    sh_9_4 = (1/306)*math.sqrt(58786)*sh_8_12*y + (1/153)*math.sqrt(969)*sh_8_14*y - 1/153*math.sqrt(969)*sh_8_2*x + (2/153)*math.sqrt(4522)*sh_8_3*z + (1/306)*math.sqrt(58786)*sh_8_4*x
    sh_9_5 = (1/153)*math.sqrt(12597)*sh_8_11*y + (1/153)*math.sqrt(1615)*sh_8_13*y - 1/153*math.sqrt(1615)*sh_8_3*x + (1/153)*math.sqrt(20995)*sh_8_4*z + (1/153)*math.sqrt(12597)*sh_8_5*x
    sh_9_6 = (1/153)*math.sqrt(10659)*sh_8_10*y + (1/306)*math.sqrt(9690)*sh_8_12*y - 1/306*math.sqrt(9690)*sh_8_4*x + (2/51)*math.sqrt(646)*sh_8_5*z + (1/153)*math.sqrt(10659)*sh_8_6*x
    sh_9_7 = (1/306)*math.sqrt(13566)*sh_8_11*y - 1/306*math.sqrt(13566)*sh_8_5*x + (1/153)*math.sqrt(24871)*sh_8_6*z + (1/306)*math.sqrt(35530)*sh_8_7*x + (1/306)*math.sqrt(35530)*sh_8_9*y
    sh_9_8 = (1/153)*math.sqrt(4522)*sh_8_10*y - 1/153*math.sqrt(4522)*sh_8_6*x + (4/153)*math.sqrt(1615)*sh_8_7*z + (1/51)*math.sqrt(1615)*sh_8_8*y
    sh_9_9 = (1/51)*math.sqrt(323)*(-2*sh_8_7*y + 3*sh_8_8*z - 2*sh_8_9*x)
    sh_9_10 = -1/153*math.sqrt(4522)*sh_8_10*x - 1/153*math.sqrt(4522)*sh_8_6*y + (1/51)*math.sqrt(1615)*sh_8_8*x + (4/153)*math.sqrt(1615)*sh_8_9*z
    sh_9_11 = (1/153)*math.sqrt(24871)*sh_8_10*z - 1/306*math.sqrt(13566)*sh_8_11*x - 1/306*math.sqrt(13566)*sh_8_5*y - 1/306*math.sqrt(35530)*sh_8_7*y + (1/306)*math.sqrt(35530)*sh_8_9*x
    sh_9_12 = (1/153)*math.sqrt(10659)*sh_8_10*x + (2/51)*math.sqrt(646)*sh_8_11*z - 1/306*math.sqrt(9690)*sh_8_12*x - 1/306*math.sqrt(9690)*sh_8_4*y - 1/153*math.sqrt(10659)*sh_8_6*y
    sh_9_13 = (1/153)*math.sqrt(12597)*sh_8_11*x + (1/153)*math.sqrt(20995)*sh_8_12*z - 1/153*math.sqrt(1615)*sh_8_13*x - 1/153*math.sqrt(1615)*sh_8_3*y - 1/153*math.sqrt(12597)*sh_8_5*y
    sh_9_14 = (1/306)*math.sqrt(58786)*sh_8_12*x + (2/153)*math.sqrt(4522)*sh_8_13*z - 1/153*math.sqrt(969)*sh_8_14*x - 1/153*math.sqrt(969)*sh_8_2*y - 1/306*math.sqrt(58786)*sh_8_4*y
    sh_9_15 = -1/306*math.sqrt(1938)*sh_8_1*y + (1/306)*math.sqrt(67830)*sh_8_13*x + (1/51)*math.sqrt(1615)*sh_8_14*z - 1/306*math.sqrt(1938)*sh_8_15*x - 1/306*math.sqrt(67830)*sh_8_3*y
    sh_9_16 = -1/306*math.sqrt(646)*sh_8_0*y + (2/153)*math.sqrt(4845)*sh_8_14*x + (4/153)*math.sqrt(646)*sh_8_15*z - 1/306*math.sqrt(646)*sh_8_16*x - 2/153*math.sqrt(4845)*sh_8_2*y
    sh_9_17 = (1/9)*math.sqrt(19)*(-2*sh_8_1*y + 2*sh_8_15*x + sh_8_16*z)
    sh_9_18 = (1/6)*math.sqrt(38)*(-sh_8_0*y + sh_8_16*x)
    if lmax == 9:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18
        ], dim=dim)

    sh_10_0 = (1/10)*math.sqrt(105)*(sh_9_0*x + sh_9_18*y)
    sh_10_1 = (1/10)*math.sqrt(21)*sh_9_0*z + (3/20)*math.sqrt(42)*sh_9_1*x + (3/20)*math.sqrt(42)*sh_9_17*y
    sh_10_2 = -1/380*math.sqrt(798)*sh_9_0*x + (3/95)*math.sqrt(399)*sh_9_1*z + (3/380)*math.sqrt(13566)*sh_9_16*y + (1/380)*math.sqrt(798)*sh_9_18*y + (3/380)*math.sqrt(13566)*sh_9_2*x
    sh_10_3 = -3/380*math.sqrt(266)*sh_9_1*x + (1/95)*math.sqrt(6783)*sh_9_15*y + (3/380)*math.sqrt(266)*sh_9_17*y + (3/190)*math.sqrt(2261)*sh_9_2*z + (1/95)*math.sqrt(6783)*sh_9_3*x
    sh_10_4 = (3/95)*math.sqrt(665)*sh_9_14*y + (3/190)*math.sqrt(133)*sh_9_16*y - 3/190*math.sqrt(133)*sh_9_2*x + (4/95)*math.sqrt(399)*sh_9_3*z + (3/95)*math.sqrt(665)*sh_9_4*x
    sh_10_5 = (21/380)*math.sqrt(190)*sh_9_13*y + (1/190)*math.sqrt(1995)*sh_9_15*y - 1/190*math.sqrt(1995)*sh_9_3*x + (3/38)*math.sqrt(133)*sh_9_4*z + (21/380)*math.sqrt(190)*sh_9_5*x
    sh_10_6 = (7/380)*math.sqrt(1482)*sh_9_12*y + (3/380)*math.sqrt(1330)*sh_9_14*y - 3/380*math.sqrt(1330)*sh_9_4*x + (21/95)*math.sqrt(19)*sh_9_5*z + (7/380)*math.sqrt(1482)*sh_9_6*x
    sh_10_7 = (3/190)*math.sqrt(1729)*sh_9_11*y + (21/380)*math.sqrt(38)*sh_9_13*y - 21/380*math.sqrt(38)*sh_9_5*x + (7/190)*math.sqrt(741)*sh_9_6*z + (3/190)*math.sqrt(1729)*sh_9_7*x
    sh_10_8 = (3/190)*math.sqrt(1463)*sh_9_10*y + (7/190)*math.sqrt(114)*sh_9_12*y - 7/190*math.sqrt(114)*sh_9_6*x + (6/95)*math.sqrt(266)*sh_9_7*z + (3/190)*math.sqrt(1463)*sh_9_8*x
    sh_10_9 = (3/190)*math.sqrt(798)*sh_9_11*y - 3/190*math.sqrt(798)*sh_9_7*x + (3/190)*math.sqrt(4389)*sh_9_8*z + (1/190)*math.sqrt(21945)*sh_9_9*y
    sh_10_10 = -3/190*math.sqrt(1995)*sh_9_10*x - 3/190*math.sqrt(1995)*sh_9_8*y + (1/19)*math.sqrt(399)*sh_9_9*z
    sh_10_11 = (3/190)*math.sqrt(4389)*sh_9_10*z - 3/190*math.sqrt(798)*sh_9_11*x - 3/190*math.sqrt(798)*sh_9_7*y + (1/190)*math.sqrt(21945)*sh_9_9*x
    sh_10_12 = (3/190)*math.sqrt(1463)*sh_9_10*x + (6/95)*math.sqrt(266)*sh_9_11*z - 7/190*math.sqrt(114)*sh_9_12*x - 7/190*math.sqrt(114)*sh_9_6*y - 3/190*math.sqrt(1463)*sh_9_8*y
    sh_10_13 = (3/190)*math.sqrt(1729)*sh_9_11*x + (7/190)*math.sqrt(741)*sh_9_12*z - 21/380*math.sqrt(38)*sh_9_13*x - 21/380*math.sqrt(38)*sh_9_5*y - 3/190*math.sqrt(1729)*sh_9_7*y
    sh_10_14 = (7/380)*math.sqrt(1482)*sh_9_12*x + (21/95)*math.sqrt(19)*sh_9_13*z - 3/380*math.sqrt(1330)*sh_9_14*x - 3/380*math.sqrt(1330)*sh_9_4*y - 7/380*math.sqrt(1482)*sh_9_6*y
    sh_10_15 = (21/380)*math.sqrt(190)*sh_9_13*x + (3/38)*math.sqrt(133)*sh_9_14*z - 1/190*math.sqrt(1995)*sh_9_15*x - 1/190*math.sqrt(1995)*sh_9_3*y - 21/380*math.sqrt(190)*sh_9_5*y
    sh_10_16 = (3/95)*math.sqrt(665)*sh_9_14*x + (4/95)*math.sqrt(399)*sh_9_15*z - 3/190*math.sqrt(133)*sh_9_16*x - 3/190*math.sqrt(133)*sh_9_2*y - 3/95*math.sqrt(665)*sh_9_4*y
    sh_10_17 = -3/380*math.sqrt(266)*sh_9_1*y + (1/95)*math.sqrt(6783)*sh_9_15*x + (3/190)*math.sqrt(2261)*sh_9_16*z - 3/380*math.sqrt(266)*sh_9_17*x - 1/95*math.sqrt(6783)*sh_9_3*y
    sh_10_18 = -1/380*math.sqrt(798)*sh_9_0*y + (3/380)*math.sqrt(13566)*sh_9_16*x + (3/95)*math.sqrt(399)*sh_9_17*z - 1/380*math.sqrt(798)*sh_9_18*x - 3/380*math.sqrt(13566)*sh_9_2*y
    sh_10_19 = -3/20*math.sqrt(42)*sh_9_1*y + (3/20)*math.sqrt(42)*sh_9_17*x + (1/10)*math.sqrt(21)*sh_9_18*z
    sh_10_20 = (1/10)*math.sqrt(105)*(-sh_9_0*y + sh_9_18*x)
    if lmax == 10:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20
        ], dim=dim)

    sh_11_0 = (1/22)*math.sqrt(506)*(sh_10_0*x + sh_10_20*y)
    sh_11_1 = (1/11)*math.sqrt(23)*sh_10_0*z + (1/11)*math.sqrt(115)*sh_10_1*x + (1/11)*math.sqrt(115)*sh_10_19*y
    sh_11_2 = -1/462*math.sqrt(966)*sh_10_0*x + (2/231)*math.sqrt(4830)*sh_10_1*z + (1/231)*math.sqrt(45885)*sh_10_18*y + (1/231)*math.sqrt(45885)*sh_10_2*x + (1/462)*math.sqrt(966)*sh_10_20*y
    sh_11_3 = -1/154*math.sqrt(322)*sh_10_1*x + (1/154)*math.sqrt(18354)*sh_10_17*y + (1/154)*math.sqrt(322)*sh_10_19*y + (1/77)*math.sqrt(3059)*sh_10_2*z + (1/154)*math.sqrt(18354)*sh_10_3*x
    sh_11_4 = (1/154)*math.sqrt(16422)*sh_10_16*y + (1/77)*math.sqrt(161)*sh_10_18*y - 1/77*math.sqrt(161)*sh_10_2*x + (2/77)*math.sqrt(966)*sh_10_3*z + (1/154)*math.sqrt(16422)*sh_10_4*x
    sh_11_5 = (2/231)*math.sqrt(8211)*sh_10_15*y + (1/231)*math.sqrt(2415)*sh_10_17*y - 1/231*math.sqrt(2415)*sh_10_3*x + (1/231)*math.sqrt(41055)*sh_10_4*z + (2/231)*math.sqrt(8211)*sh_10_5*x
    sh_11_6 = (2/77)*math.sqrt(805)*sh_10_14*y + (1/154)*math.sqrt(1610)*sh_10_16*y - 1/154*math.sqrt(1610)*sh_10_4*x + (4/77)*math.sqrt(322)*sh_10_5*z + (2/77)*math.sqrt(805)*sh_10_6*x
    sh_11_7 = (1/22)*math.sqrt(230)*sh_10_13*y + (1/22)*math.sqrt(46)*sh_10_15*y - 1/22*math.sqrt(46)*sh_10_5*x + (1/11)*math.sqrt(115)*sh_10_6*z + (1/22)*math.sqrt(230)*sh_10_7*x
    sh_11_8 = (1/66)*math.sqrt(1794)*sh_10_12*y + (1/33)*math.sqrt(138)*sh_10_14*y - 1/33*math.sqrt(138)*sh_10_6*x + (4/33)*math.sqrt(69)*sh_10_7*z + (1/66)*math.sqrt(1794)*sh_10_8*x
    sh_11_9 = (1/77)*math.sqrt(2093)*sh_10_11*y + (1/77)*math.sqrt(966)*sh_10_13*y - 1/77*math.sqrt(966)*sh_10_7*x + (1/77)*math.sqrt(6279)*sh_10_8*z + (1/77)*math.sqrt(2093)*sh_10_9*x
    sh_11_10 = (1/77)*math.sqrt(3542)*sh_10_10*y + (1/154)*math.sqrt(4830)*sh_10_12*y - 1/154*math.sqrt(4830)*sh_10_8*x + (2/77)*math.sqrt(1610)*sh_10_9*z
    sh_11_11 = (1/21)*math.sqrt(483)*sh_10_10*z - 1/231*math.sqrt(26565)*sh_10_11*x - 1/231*math.sqrt(26565)*sh_10_9*y
    sh_11_12 = (1/77)*math.sqrt(3542)*sh_10_10*x + (2/77)*math.sqrt(1610)*sh_10_11*z - 1/154*math.sqrt(4830)*sh_10_12*x - 1/154*math.sqrt(4830)*sh_10_8*y
    sh_11_13 = (1/77)*math.sqrt(2093)*sh_10_11*x + (1/77)*math.sqrt(6279)*sh_10_12*z - 1/77*math.sqrt(966)*sh_10_13*x - 1/77*math.sqrt(966)*sh_10_7*y - 1/77*math.sqrt(2093)*sh_10_9*y
    sh_11_14 = (1/66)*math.sqrt(1794)*sh_10_12*x + (4/33)*math.sqrt(69)*sh_10_13*z - 1/33*math.sqrt(138)*sh_10_14*x - 1/33*math.sqrt(138)*sh_10_6*y - 1/66*math.sqrt(1794)*sh_10_8*y
    sh_11_15 = (1/22)*math.sqrt(230)*sh_10_13*x + (1/11)*math.sqrt(115)*sh_10_14*z - 1/22*math.sqrt(46)*sh_10_15*x - 1/22*math.sqrt(46)*sh_10_5*y - 1/22*math.sqrt(230)*sh_10_7*y
    sh_11_16 = (2/77)*math.sqrt(805)*sh_10_14*x + (4/77)*math.sqrt(322)*sh_10_15*z - 1/154*math.sqrt(1610)*sh_10_16*x - 1/154*math.sqrt(1610)*sh_10_4*y - 2/77*math.sqrt(805)*sh_10_6*y
    sh_11_17 = (2/231)*math.sqrt(8211)*sh_10_15*x + (1/231)*math.sqrt(41055)*sh_10_16*z - 1/231*math.sqrt(2415)*sh_10_17*x - 1/231*math.sqrt(2415)*sh_10_3*y - 2/231*math.sqrt(8211)*sh_10_5*y
    sh_11_18 = (1/154)*math.sqrt(16422)*sh_10_16*x + (2/77)*math.sqrt(966)*sh_10_17*z - 1/77*math.sqrt(161)*sh_10_18*x - 1/77*math.sqrt(161)*sh_10_2*y - 1/154*math.sqrt(16422)*sh_10_4*y
    sh_11_19 = -1/154*math.sqrt(322)*sh_10_1*y + (1/154)*math.sqrt(18354)*sh_10_17*x + (1/77)*math.sqrt(3059)*sh_10_18*z - 1/154*math.sqrt(322)*sh_10_19*x - 1/154*math.sqrt(18354)*sh_10_3*y
    sh_11_20 = -1/462*math.sqrt(966)*sh_10_0*y + (1/231)*math.sqrt(45885)*sh_10_18*x + (2/231)*math.sqrt(4830)*sh_10_19*z - 1/231*math.sqrt(45885)*sh_10_2*y - 1/462*math.sqrt(966)*sh_10_20*x
    sh_11_21 = -1/11*math.sqrt(115)*sh_10_1*y + (1/11)*math.sqrt(115)*sh_10_19*x + (1/11)*math.sqrt(23)*sh_10_20*z
    sh_11_22 = (1/22)*math.sqrt(506)*(-sh_10_0*y + sh_10_20*x)
    if lmax == 11:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22
        ], dim=dim)

    sh_12_0 = (5/12)*math.sqrt(6)*(sh_11_0*x + sh_11_22*y)
    sh_12_1 = (5/12)*sh_11_0*z + (5/24)*math.sqrt(22)*sh_11_1*x + (5/24)*math.sqrt(22)*sh_11_21*y
    sh_12_2 = -5/552*math.sqrt(46)*sh_11_0*x + (5/138)*math.sqrt(253)*sh_11_1*z + (5/552)*math.sqrt(10626)*sh_11_2*x + (5/552)*math.sqrt(10626)*sh_11_20*y + (5/552)*math.sqrt(46)*sh_11_22*y
    sh_12_3 = -5/552*math.sqrt(138)*sh_11_1*x + (5/276)*math.sqrt(2415)*sh_11_19*y + (5/92)*math.sqrt(161)*sh_11_2*z + (5/552)*math.sqrt(138)*sh_11_21*y + (5/276)*math.sqrt(2415)*sh_11_3*x
    sh_12_4 = (5/276)*math.sqrt(2185)*sh_11_18*y - 5/276*math.sqrt(69)*sh_11_2*x + (5/276)*math.sqrt(69)*sh_11_20*y + (5/69)*math.sqrt(115)*sh_11_3*z + (5/276)*math.sqrt(2185)*sh_11_4*x
    sh_12_5 = (5/184)*math.sqrt(874)*sh_11_17*y + (5/276)*math.sqrt(115)*sh_11_19*y - 5/276*math.sqrt(115)*sh_11_3*x + (5/276)*math.sqrt(2185)*sh_11_4*z + (5/184)*math.sqrt(874)*sh_11_5*x
    sh_12_6 = (5/552)*math.sqrt(3)*(math.sqrt(2346)*sh_11_16*y + math.sqrt(230)*sh_11_18*y - math.sqrt(230)*sh_11_4*x + 12*math.sqrt(23)*sh_11_5*z + math.sqrt(2346)*sh_11_6*x)
    sh_12_7 = (5/138)*math.sqrt(391)*sh_11_15*y + (5/552)*math.sqrt(966)*sh_11_17*y - 5/552*math.sqrt(966)*sh_11_5*x + (5/276)*math.sqrt(2737)*sh_11_6*z + (5/138)*math.sqrt(391)*sh_11_7*x
    sh_12_8 = (5/138)*math.sqrt(345)*sh_11_14*y + (5/276)*math.sqrt(322)*sh_11_16*y - 5/276*math.sqrt(322)*sh_11_6*x + (10/69)*math.sqrt(46)*sh_11_7*z + (5/138)*math.sqrt(345)*sh_11_8*x
    sh_12_9 = (5/552)*math.sqrt(4830)*sh_11_13*y + (5/92)*math.sqrt(46)*sh_11_15*y - 5/92*math.sqrt(46)*sh_11_7*x + (5/92)*math.sqrt(345)*sh_11_8*z + (5/552)*math.sqrt(4830)*sh_11_9*x
    sh_12_10 = (5/552)*math.sqrt(4186)*sh_11_10*x + (5/552)*math.sqrt(4186)*sh_11_12*y + (5/184)*math.sqrt(230)*sh_11_14*y - 5/184*math.sqrt(230)*sh_11_8*x + (5/138)*math.sqrt(805)*sh_11_9*z
    sh_12_11 = (5/276)*math.sqrt(3289)*sh_11_10*z + (5/276)*math.sqrt(1794)*sh_11_11*y + (5/552)*math.sqrt(2530)*sh_11_13*y - 5/552*math.sqrt(2530)*sh_11_9*x
    sh_12_12 = -5/276*math.sqrt(1518)*sh_11_10*y + (5/23)*math.sqrt(23)*sh_11_11*z - 5/276*math.sqrt(1518)*sh_11_12*x
    sh_12_13 = (5/276)*math.sqrt(1794)*sh_11_11*x + (5/276)*math.sqrt(3289)*sh_11_12*z - 5/552*math.sqrt(2530)*sh_11_13*x - 5/552*math.sqrt(2530)*sh_11_9*y
    sh_12_14 = -5/552*math.sqrt(4186)*sh_11_10*y + (5/552)*math.sqrt(4186)*sh_11_12*x + (5/138)*math.sqrt(805)*sh_11_13*z - 5/184*math.sqrt(230)*sh_11_14*x - 5/184*math.sqrt(230)*sh_11_8*y
    sh_12_15 = (5/552)*math.sqrt(4830)*sh_11_13*x + (5/92)*math.sqrt(345)*sh_11_14*z - 5/92*math.sqrt(46)*sh_11_15*x - 5/92*math.sqrt(46)*sh_11_7*y - 5/552*math.sqrt(4830)*sh_11_9*y
    sh_12_16 = (5/138)*math.sqrt(345)*sh_11_14*x + (10/69)*math.sqrt(46)*sh_11_15*z - 5/276*math.sqrt(322)*sh_11_16*x - 5/276*math.sqrt(322)*sh_11_6*y - 5/138*math.sqrt(345)*sh_11_8*y
    sh_12_17 = (5/138)*math.sqrt(391)*sh_11_15*x + (5/276)*math.sqrt(2737)*sh_11_16*z - 5/552*math.sqrt(966)*sh_11_17*x - 5/552*math.sqrt(966)*sh_11_5*y - 5/138*math.sqrt(391)*sh_11_7*y
    sh_12_18 = (5/552)*math.sqrt(3)*(math.sqrt(2346)*sh_11_16*x + 12*math.sqrt(23)*sh_11_17*z - math.sqrt(230)*sh_11_18*x - math.sqrt(230)*sh_11_4*y - math.sqrt(2346)*sh_11_6*y)
    sh_12_19 = (5/184)*math.sqrt(874)*sh_11_17*x + (5/276)*math.sqrt(2185)*sh_11_18*z - 5/276*math.sqrt(115)*sh_11_19*x - 5/276*math.sqrt(115)*sh_11_3*y - 5/184*math.sqrt(874)*sh_11_5*y
    sh_12_20 = (5/276)*math.sqrt(2185)*sh_11_18*x + (5/69)*math.sqrt(115)*sh_11_19*z - 5/276*math.sqrt(69)*sh_11_2*y - 5/276*math.sqrt(69)*sh_11_20*x - 5/276*math.sqrt(2185)*sh_11_4*y
    sh_12_21 = -5/552*math.sqrt(138)*sh_11_1*y + (5/276)*math.sqrt(2415)*sh_11_19*x + (5/92)*math.sqrt(161)*sh_11_20*z - 5/552*math.sqrt(138)*sh_11_21*x - 5/276*math.sqrt(2415)*sh_11_3*y
    sh_12_22 = -5/552*math.sqrt(46)*sh_11_0*y - 5/552*math.sqrt(10626)*sh_11_2*y + (5/552)*math.sqrt(10626)*sh_11_20*x + (5/138)*math.sqrt(253)*sh_11_21*z - 5/552*math.sqrt(46)*sh_11_22*x
    sh_12_23 = -5/24*math.sqrt(22)*sh_11_1*y + (5/24)*math.sqrt(22)*sh_11_21*x + (5/12)*sh_11_22*z
    sh_12_24 = (5/12)*math.sqrt(6)*(-sh_11_0*y + sh_11_22*x)
    if lmax == 12:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24
        ], dim=dim)

    sh_13_0 = (3/26)*math.sqrt(78)*(sh_12_0*x + sh_12_24*y)
    sh_13_1 = (3/13)*math.sqrt(3)*sh_12_0*z + (9/13)*math.sqrt(2)*sh_12_1*x + (9/13)*math.sqrt(2)*sh_12_23*y
    sh_13_2 = -3/130*math.sqrt(6)*sh_12_0*x + (36/65)*sh_12_1*z + (9/65)*math.sqrt(46)*sh_12_2*x + (9/65)*math.sqrt(46)*sh_12_22*y + (3/130)*math.sqrt(6)*sh_12_24*y
    sh_13_3 = -9/130*math.sqrt(2)*sh_12_1*x + (9/65)*math.sqrt(23)*sh_12_2*z + (3/130)*math.sqrt(1518)*sh_12_21*y + (9/130)*math.sqrt(2)*sh_12_23*y + (3/130)*math.sqrt(1518)*sh_12_3*x
    sh_13_4 = -9/65*sh_12_2*x + (9/130)*math.sqrt(154)*sh_12_20*y + (9/65)*sh_12_22*y + (6/65)*math.sqrt(66)*sh_12_3*z + (9/130)*math.sqrt(154)*sh_12_4*x
    sh_13_5 = (9/65)*math.sqrt(35)*sh_12_19*y + (3/65)*math.sqrt(15)*sh_12_21*y - 3/65*math.sqrt(15)*sh_12_3*x + (9/65)*math.sqrt(35)*sh_12_4*z + (9/65)*math.sqrt(35)*sh_12_5*x
    sh_13_6 = (3/65)*math.sqrt(285)*sh_12_18*y + (9/130)*math.sqrt(10)*sh_12_20*y - 9/130*math.sqrt(10)*sh_12_4*x + (18/65)*math.sqrt(10)*sh_12_5*z + (3/65)*math.sqrt(285)*sh_12_6*x
    sh_13_7 = (9/130)*math.sqrt(114)*sh_12_17*y + (9/130)*math.sqrt(14)*sh_12_19*y - 9/130*math.sqrt(14)*sh_12_5*x + (3/65)*math.sqrt(399)*sh_12_6*z + (9/130)*math.sqrt(114)*sh_12_7*x
    sh_13_8 = (9/130)*math.sqrt(102)*sh_12_16*y + (3/65)*math.sqrt(42)*sh_12_18*y - 3/65*math.sqrt(42)*sh_12_6*x + (36/65)*math.sqrt(3)*sh_12_7*z + (9/130)*math.sqrt(102)*sh_12_8*x
    sh_13_9 = (6/65)*math.sqrt(51)*sh_12_15*y + (9/65)*math.sqrt(6)*sh_12_17*y - 9/65*math.sqrt(6)*sh_12_7*x + (9/65)*math.sqrt(51)*sh_12_8*z + (6/65)*math.sqrt(51)*sh_12_9*x
    sh_13_10 = (18/65)*math.sqrt(5)*sh_12_10*x + (18/65)*math.sqrt(5)*sh_12_14*y + (9/130)*math.sqrt(30)*sh_12_16*y - 9/130*math.sqrt(30)*sh_12_8*x + (12/65)*math.sqrt(30)*sh_12_9*z
    sh_13_11 = (9/65)*math.sqrt(55)*sh_12_10*z + (9/130)*math.sqrt(70)*sh_12_11*x + (9/130)*math.sqrt(70)*sh_12_13*y + (3/130)*math.sqrt(330)*sh_12_15*y - 3/130*math.sqrt(330)*sh_12_9*x
    sh_13_12 = -9/65*math.sqrt(11)*sh_12_10*x + (18/65)*math.sqrt(14)*sh_12_11*z + (3/65)*math.sqrt(273)*sh_12_12*y + (9/65)*math.sqrt(11)*sh_12_14*y
    sh_13_13 = -9/65*math.sqrt(26)*sh_12_11*y + (3/5)*math.sqrt(3)*sh_12_12*z - 9/65*math.sqrt(26)*sh_12_13*x
    sh_13_14 = -9/65*math.sqrt(11)*sh_12_10*y + (3/65)*math.sqrt(273)*sh_12_12*x + (18/65)*math.sqrt(14)*sh_12_13*z - 9/65*math.sqrt(11)*sh_12_14*x
    sh_13_15 = -9/130*math.sqrt(70)*sh_12_11*y + (9/130)*math.sqrt(70)*sh_12_13*x + (9/65)*math.sqrt(55)*sh_12_14*z - 3/130*math.sqrt(330)*sh_12_15*x - 3/130*math.sqrt(330)*sh_12_9*y
    sh_13_16 = -18/65*math.sqrt(5)*sh_12_10*y + (18/65)*math.sqrt(5)*sh_12_14*x + (12/65)*math.sqrt(30)*sh_12_15*z - 9/130*math.sqrt(30)*sh_12_16*x - 9/130*math.sqrt(30)*sh_12_8*y
    sh_13_17 = (6/65)*math.sqrt(51)*sh_12_15*x + (9/65)*math.sqrt(51)*sh_12_16*z - 9/65*math.sqrt(6)*sh_12_17*x - 9/65*math.sqrt(6)*sh_12_7*y - 6/65*math.sqrt(51)*sh_12_9*y
    sh_13_18 = (9/130)*math.sqrt(102)*sh_12_16*x + (36/65)*math.sqrt(3)*sh_12_17*z - 3/65*math.sqrt(42)*sh_12_18*x - 3/65*math.sqrt(42)*sh_12_6*y - 9/130*math.sqrt(102)*sh_12_8*y
    sh_13_19 = (9/130)*math.sqrt(114)*sh_12_17*x + (3/65)*math.sqrt(399)*sh_12_18*z - 9/130*math.sqrt(14)*sh_12_19*x - 9/130*math.sqrt(14)*sh_12_5*y - 9/130*math.sqrt(114)*sh_12_7*y
    sh_13_20 = (3/65)*math.sqrt(285)*sh_12_18*x + (18/65)*math.sqrt(10)*sh_12_19*z - 9/130*math.sqrt(10)*sh_12_20*x - 9/130*math.sqrt(10)*sh_12_4*y - 3/65*math.sqrt(285)*sh_12_6*y
    sh_13_21 = (9/65)*math.sqrt(35)*sh_12_19*x + (9/65)*math.sqrt(35)*sh_12_20*z - 3/65*math.sqrt(15)*sh_12_21*x - 3/65*math.sqrt(15)*sh_12_3*y - 9/65*math.sqrt(35)*sh_12_5*y
    sh_13_22 = -9/65*sh_12_2*y + (9/130)*math.sqrt(154)*sh_12_20*x + (6/65)*math.sqrt(66)*sh_12_21*z - 9/65*sh_12_22*x - 9/130*math.sqrt(154)*sh_12_4*y
    sh_13_23 = -9/130*math.sqrt(2)*sh_12_1*y + (3/130)*math.sqrt(1518)*sh_12_21*x + (9/65)*math.sqrt(23)*sh_12_22*z - 9/130*math.sqrt(2)*sh_12_23*x - 3/130*math.sqrt(1518)*sh_12_3*y
    sh_13_24 = -3/130*math.sqrt(6)*sh_12_0*y - 9/65*math.sqrt(46)*sh_12_2*y + (9/65)*math.sqrt(46)*sh_12_22*x + (36/65)*sh_12_23*z - 3/130*math.sqrt(6)*sh_12_24*x
    sh_13_25 = -9/13*math.sqrt(2)*sh_12_1*y + (9/13)*math.sqrt(2)*sh_12_23*x + (3/13)*math.sqrt(3)*sh_12_24*z
    sh_13_26 = (3/26)*math.sqrt(78)*(-sh_12_0*y + sh_12_24*x)
    if lmax == 13:
        return torch.cat([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
            sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24,
            sh_13_0, sh_13_1, sh_13_2, sh_13_3, sh_13_4, sh_13_5, sh_13_6, sh_13_7, sh_13_8, sh_13_9, sh_13_10, sh_13_11, sh_13_12, sh_13_13, sh_13_14, sh_13_15, sh_13_16, sh_13_17, sh_13_18, sh_13_19, sh_13_20, sh_13_21, sh_13_22, sh_13_23, sh_13_24, sh_13_25, sh_13_26
        ], dim=dim)

