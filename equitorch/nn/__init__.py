"""
Some neural network layers.
"""
from .linear import *
from .tensor_product import *
from .activation import *
from .attention import *
from .radial_basis import *
from .angular_basis import *
from .spherical_basis import *
from .cutoff import *
from .normalization import *
from .other import *

linears = [
    'SO3Linear',
    'SO2Linear',
    'DegreeWiseLinear',
]

cutoffs = [
    'CosineCutoff',
    'MollifierCutoff',
    'PolynomialCutoff',    
]

activations = [
    'Gate',
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