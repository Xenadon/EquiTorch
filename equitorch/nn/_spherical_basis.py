from torch import Tensor, nn

from ..math._o3 import spherical_harmonics
from ..typing import DegreeRange
from ..utils._indices import check_degree_range

class SphericalHarmonicBasisExpansion(nn.Module):
    r"""
    Expand Cartesian vectors on a sphere to spherical harmonics.

    .. math::
        Y_m^{(l)}(\hat{\mathbf{r}}),\ l\in L;\ m=-l,\dots,l.

    See also: :obj:`~math.spherical_harmonics` for details about 
    spherical harmonics.

    Parameters
    ----------
    L : DegreeRange
        The degree range of the spherical harmonics.
    """
    def __init__(self, 
                 L: DegreeRange,
                ):
        super().__init__()
        self.L = check_degree_range(L)

    def forward(self, x: Tensor):
        """
        Parameters
        ----------
        x : Tensor
            The Cartesian vectors on a sphere.

        Returns
        -------
        Tensor
            The spherical harmonic encoding.        
        """
        return spherical_harmonics(x, self.L)