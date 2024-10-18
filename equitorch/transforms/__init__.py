"""
Some data transforms.
"""
from typing import Callable
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from ..utils.indices import check_degree_range

from ..utils.geometries import align_to_z_mat, align_to_z_wigner, edge_align_to_z_mat

from ..math.so3 import spherical_harmonics

from ..typing import DegreeRange


# Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/radius_graph.html#RadiusGraph
@functional_transform('radius_graph_et')
class RadiusGraph(BaseTransform):
    r"""Creates edges based on node positions :obj:`pos_attr` to all points
    within a given cutoff distance (functional name: :obj:`radius_graph_et`).

    Parameters
    ----------
    r : float
        The cutoff distance.
    loop : bool, optional
        If True, the graph will contain self-loops. Default is :obj:`False`.
    max_num_neighbors : int, optional
        The maximum number of neighbors to return for each element.
        This flag is only needed for CUDA tensors. Default is 32.
    flow : str, optional
        The flow direction when using in combination with message passing
        ("source_to_target" or "target_to_source"). Default is "source_to_target".
    pos_attr : str, optional
        The attribute name for positions in the data. Default is "pos".
    edge_index_attr : str, optional
        The attribute name for creating edge index in the data. Default is "edge_index".
    edge_vector_attr : str, optional
        The attribute name for creating edge vectors in the data. Default is "edge_vec".
    num_workers : int, optional
        Number of workers to use for computation. Has no effect in case batch is
        not None, or the input lies on the GPU. Default is 1.

    Example
    --------
    >>> N = 50
    >>> pos = torch.randn(N,3)
    >>> data = Data(pos=pos)
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
    >>> data = RadiusGraph(0.5)(data)
    >>> print(data)
    Data(pos=[50, 3])
    """
    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        pos_attr: str = 'pos',
        edge_index_attr: str = 'edge_index',
        edge_vector_attr: str = 'edge_vec',
        num_workers: int = 1,
    ) -> None:
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

        self.pos_attr = pos_attr
        self.edge_index_attr = edge_index_attr
        self.edge_vector_attr = edge_vector_attr

    def forward(self, data: Data) -> Data:
        # assert data.pos is not None


        pos = data.__getattr__(self.pos_attr)
        edge_index = torch_geometric.nn.radius_graph(
            data.pos,
            self.r,
            data.batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )
        data.__setattr__(self.edge_index_attr, edge_index)

        data.__setattr__(self.edge_vector_attr, 
            data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]]
        )

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'
    
    
@functional_transform('add_edge_spherical_harmonics')
class AddEdgeSphericalHarmonics(BaseTransform):
    r"""Creates edge spherical harmonics embedding
    based on edge direction vectors :obj:`edge_vector_attr`  
    (functional name: :obj:`add_edge_spherical_harmonics`).

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range of spherical harmonics.
    edge_vector_attr : str, optional
        The attribute name for edge direction vectors. Default is "edge_vec".
    edge_sh_attr : str, optional
        The attribute name for creating edge spherical harmonics in the data. Default is "edge_sh".
    
    Example
    --------
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
    >>> data = AddEdgeSphericalHarmonics(L=3)(data)
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3], edge_sh=[36, 16])
    """
    def __init__(
        self,
        L: DegreeRange,
        edge_vector_attr: str = 'edge_vec',
        edge_sh_attr: str = 'edge_sh'
    ) -> None:

        self.L = check_degree_range(L)
        self.edge_vector_attr = edge_vector_attr
        self.edge_sh_attr = edge_sh_attr

    def forward(self, data: Data) -> Data:
        
        edge_vec = data.__getattr__(self.edge_vector_attr)
        data.__setattr__(self.edge_sh_attr, spherical_harmonics(edge_vec, self.L, True, dim=-1))

        return data

@functional_transform('add_edge_align_matrix')   
class AddEdgeAlignMatrix(BaseTransform):
    r"""Creates rotation matrices that can align each edge to z
    based on edge direction vectors :obj:`edge_vector_attr` 
    (functional name: :obj:`add_edge_align_matrix`).

    Parameters
    ----------
    edge_vector_attr : str, optional
        The attribute name for edge direction vectors. Default is "edge_vec".
    align_mat_attr : str, optional
        The attribute name for creating edge alignment matrices in the data. Default is "R".

    Example
    --------
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
    >>> data = AddEdgeAlignMatrix()(data)
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3], R=[36, 3, 3])
    """
    def __init__(
        self,
        edge_vector_attr: str = 'edge_vec',
        align_mat_attr: str = 'R'
    ) -> None:
        self.edge_vector_attr = edge_vector_attr
        self.align_mat_attr = align_mat_attr

    def forward(self, data: Data) -> Data:
        
        edge_vec = data.__getattr__(self.edge_vector_attr)
        data.__setattr__(self.align_mat_attr, align_to_z_mat(edge_vec))

        return data

@functional_transform('add_edge_align_wigner_d')   
class AddEdgeAlignWignerD(BaseTransform):
    r"""Creates Wigner D matrices for the roation matrices that
    can align each edge to z based on edge direction vectors 
    :obj:`edge_vector_attr`. (functional name: :obj:`add_edge_align_matrix`).

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range for the Wigner D matrices.
    edge_vector_attr : str, optional
        The attribute name for edge direction vectors. Default is "edge_vec".
    align_wigner_attr : str, optional
        The attribute name for creating edge alignment matrices in the data. Default is "D".
    
    Example
    --------
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
    >>> data = AddEdgeAlignWignerD(L=1)(data)
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3], D=[36, 4, 4])
    >>> data = AddEdgeAlignWignerD(L=3, align_wigner_attr='D_3')(data)
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3], D=[36, 4, 4], D_3=[36, 16, 16])
    """
    def __init__(
        self,
        L: DegreeRange,
        edge_vector_attr: str = 'edge_vec',
        align_wigner_attr: str = 'D'
    ) -> None:

        self.L = L
        self.edge_vector_attr = edge_vector_attr
        self.align_wigner_attr = align_wigner_attr

    def forward(self, data: Data) -> Data:
        
        edge_vec = data.__getattr__(self.edge_vector_attr)
        data.__setattr__(self.align_wigner_attr, align_to_z_wigner(edge_vec, self.L))

        return data
    

@functional_transform('add_edge_length_embedding')   
class AddEdgeLengthEmbedding(BaseTransform):
    r"""Add the edge length embedding for edges.

    Parameters
    ----------
    emb: Callable[[Tensor,], Tensor]
        The length embedding operation.
    
    Example
    --------
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3])
    >>> data = AddEdgeLengthEmbedding(GaussianBasisExpansion(10, 300, 0, 30))
    >>> print(data)
    Data(pos=[50, 3], edge_index=[2, 36], edge_vec=[36, 3], edge_emb=[36, 300])
    """
    def __init__(
        self,
        emb: Callable[[Tensor], Tensor],
        edge_vector_attr: str = 'edge_vec',
        edge_embedding_attr: str = 'edge_emb'
    ) -> None:

        self.emb = emb
        self.edge_vector_attr = edge_vector_attr
        self.edge_embedding_attr = edge_embedding_attr

    def forward(self, data: Data) -> Data:
        
        edge_vec = data.__getattr__(self.edge_vector_attr)
        
        data.__setattr__(self.edge_embedding_attr, self.emb(edge_vec.norm(dim=-1)))

        return data
    
    
__all__ = [
    'RadiusGraph',
    'AddEdgeAlignMatrix',
    'AddEdgeSphericalHarmonics',
    'AddEdgeAlignWignerD',
]