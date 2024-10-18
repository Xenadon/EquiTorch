from torch import Tensor, nn

from ..math.so3 import spherical_harmonics
from ..typing import DegreeRange
from ..utils.indices import check_degree_range

class SphericalHarmonicBasisExpansion(nn.Module):
    r"""
    Expand Cartesian vectors on a sphere to spherical harmonics.

    .. math::
        Y_m^{(l)}(\hat{\mathbf{r}}),\ l\in L;\ m=-l,\dots,l.

    See also: :obj:`~math.spherical_harmonics` for details about 
    spherical harmonics.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
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
        x : :obj:`~torch.Tensor`
            The Cartesian vectors on a sphere.

        Returns
        -------
        :obj:`~torch.Tensor`
            The spherical harmonic encoding.        
        """
        return spherical_harmonics(x, self.L)