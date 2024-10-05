"""
Some neural network layers.
"""
from ._linear import *
from ._tensor_product import *
from ._activation import *
from ._attention import *
from ._radial_basis import *
from ._angular_basis import *
from ._spherical_basis import *
from ._cutoff import *
from ._normalization import *
from ._other import *

linears = [
    'SO3Linear',
    'SO2Linear',
    'DegreeWiseLinear',
    'ElementWiseLinear',
]

cutoffs = [
    'CosineCutoff',
    'MollifierCutoff',
    'PolynomialCutoff',    
]

activations = [
    'S2Act',
    'NormAct'
]

radial_basis = [
    'GaussianBasisExpansion',
    'BesselBasisExpansion',
]    

angular_basis =[
    'SineBasisExpansion',
    'CosineBasisExpansion',
    'FourierBasisExpansion',
]

spherical_basis = [
    'SphericalBasisExpansion'
]

tensor_products = [
    'WeightedTensorProduct',
    'TensorProduct',
    'TensorDot',
]

attentions = [
    'SimpleDotAttention',
    'SE3TrAttention',
    'MultiheadAttentionBlock'
]

normalizations = [
    'EquivariantLayerNorm'
]

others = [
    'Separable',
    'BranchedModuleDict',
    'BranchedModuleList',
]


__all__ = (
    linears + 
    cutoffs + 
    radial_basis + 
    angular_basis +
    tensor_products +
    attentions + 
    normalizations +
    others +
    activations
)