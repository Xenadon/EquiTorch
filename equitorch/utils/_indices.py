from typing import Tuple, List
import torch
from torch import Tensor
import functools
import torch_geometric
from torch_geometric.utils import scatter
import torch_geometric.utils

from ..typing import DegreeRange

@functools.lru_cache(maxsize=None)
def check_degree_range(L: DegreeRange) -> Tuple[int,int]:
    """
    Check and normalize the degree range.

    Parameters
    ----------
    L : DegreeRange
        The input degree range. Can be an int or a tuple of two ints.

    Returns
    -------
    Tuple[int, int]
        A tuple representing the normalized degree range (l_min, l_max).

    Example
    -------
    >>> L = (1, 3)
    >>> result = check_degree_range(L)
    >>> print(result)
    (1, 3)
    >>> L = 5
    >>> result = check_degree_range(L)
    >>> print(result)
    (0, 5)
    """

    return (0, L) if isinstance(L, int) else L

@functools.lru_cache(maxsize=None)
def degrees_in_range(L: DegreeRange):
    L = check_degree_range(L)
    return range(L[0], L[1]+1)

@functools.lru_cache(maxsize=None)
def order_batch(L: DegreeRange, device=None):
    """
    Generate a tensor of degree indices for each order.

    Parameters
    ----------
    L : DegreeRange
        The degree range.
    device : torch.device, optional
        The device to place the resulting tensor on.

    Returns
    -------
    torch.Tensor
        A tensor where each element is the degree index for the corresponding order.

    Example
    -------
    >>> L = (0, 2)
    >>> result = order_batch(L)
    >>> print(result)
    tensor([0, 1, 1, 1, 2, 2, 2, 2, 2])
    """

    L = check_degree_range(L)
    ind = torch.zeros((L[1]+1)**2 - L[0]**2, device=device, dtype=torch.int64)
    acc = 0
    for l in degrees_in_range(L):
        ind[acc: acc+2*l+1] = l-L[0]
        acc += 2*l+1
    return ind

def extract_batch_ptr(keys: List[Tensor]):
    '''
    key: sorted, int
    returns: batch, ptr, key_value
    '''
    ptr = [0]
    length = keys[0].shape[0]
    for i in range(length):
        if any(k[i] != k[ptr[-1]] for k in keys):
            ptr.append(i)
    ptr.append(i+1)
    batch = torch.zeros(length, dtype=torch.long)
    for i in range(1,len(ptr)):
        batch[ptr[i-1]:ptr[i]] = i-1
    ptr = torch.tensor(ptr)
    val = [k[ptr[:-1]] for k in keys]
    return batch, ptr, val

# def 

@functools.lru_cache(maxsize=None)
def order_ptr(L: DegreeRange, dim:int=0, device=None):
    """
    Generate a tensor of pointers to the start of each degree's orders, including the next of the final elements.

    Parameters
    ----------
    L : DegreeRange
        The degree range.
    dim : int, optional
        The dimension along which to expand the result. Default is 0.
    device : torch.device, optional
        The device to place the resulting tensor on.

    Returns
    -------
    torch.Tensor
        A tensor of pointers to the start of each degree's orders.

    Example
    -------
    >>> L = (0, 2)
    >>> result = order_ptr(L)
    >>> print(result)
    tensor([0, 1, 4, 9])
    >>> L = (1, 3)
    >>> result = order_ptr(L)
    >>> print(result)
    tensor([0, 3, 8, 15])
    """
    L = check_degree_range(L)
    return torch.tensor([i**2-L[0]**2 for i in range(L[0], L[1]+2)]).reshape([1]*dim+[-1]).to(device)

@functools.lru_cache(maxsize=None)
def list_degrees(L: DegreeRange, L1: DegreeRange, L2: DegreeRange, cond=None):
    """
    Generate a list of valid degree triplets (l, l1, l2).

    Parameters
    ----------
    L : DegreeRange
        The range for the total angular momentum l.
    L1 : DegreeRange
        The range for the first angular momentum l1.
    L2 : DegreeRange
        The range for the second angular momentum l2.
    cond : Callable[[int, int, int], bool], optional
        An optional condition function to further filter the triplets.

    Returns
    -------
    List[Tuple[int, int, int]]
        A sorted list of valid (l, l1, l2) triplets.

    Example
    -------
    >>> print(list_degrees(1, (1,2), (2,3)))
    [(0, 2, 2), (1, 1, 2), (1, 2, 2), (1, 2, 3)]
    """
    L = check_degree_range(L)
    L1 = check_degree_range(L1)
    L2 = check_degree_range(L2)
    if cond is not None:
        ls = [(l, l1, l2) 
              for l1 in degrees_in_range(L1) 
              for l2 in degrees_in_range(L2)
              for l in degrees_in_range((abs(l1-l2),l1+l2)) 
              if (l >= L[0] and l <= L[1] and cond(l, l1, l2))]
    else:
        ls = [(l, l1, l2) 
              for l1 in degrees_in_range(L1) 
              for l2 in degrees_in_range(L2)
              for l in degrees_in_range((abs(l1-l2),l1+l2))  
              if (l >= L[0] and l <= L[1])]
    return sorted(ls)

def num_order_between(lmin: int, lmax: int):
    """
    Calculate the number of orders between two degrees.

    Parameters
    ----------
    lmin : int
        The minimum degree.
    lmax : int
        The maximum degree.

    Returns
    -------
    int
        The total number of orders between lmin and lmax (inclusive).
    """
    return (lmax+1) ** 2 - lmin ** 2

def num_order_of_degree(l: int):
    """
    Calculate the number of orders for a given degree.

    Parameters
    ----------
    l : int
        The degree.

    Returns
    -------
    int
        The number of orders for the given degree (2l + 1).
    """
    return 2 * l + 1

def degree_index_from(l: int, l_min: int):
    """
    Calculate the start and end indices for a given degree.

    Parameters
    ----------
    l : int
        The degree.
    l_min : int
        The minimum degree in the range.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the start and end indices for the given degree.

    Example
    -------
    >>> l, l_min = 2, 0
    >>> result = degree_index_from(l, l_min)
    >>> print(result)
    (4, 8)
    """
    return (l**2-l_min**2) , (l**2-l_min**2)+l*2

def degree_order_to_index(l, m, l_min=0):
    """
    Convert degree and order to a linear index.

    Parameters
    ----------
    l : int
        The degree.
    m : int
        The order.
    l_min : int, optional
        The minimum degree in the range. Default is 0.

    Returns
    -------
    int
        The linear index corresponding to the given degree and order.

    Example
    -------
    >>> l, m, l_min = 2, 1, 0
    >>> result = degree_order_to_index(l, m, l_min)
    >>> print(result)
    7
    """
    return l ** 2 + m+l - l_min ** 2

def expand_degree_to_order(src: Tensor, L: DegreeRange, dim: int):
    """
    Expand a tensor from degree to order representation.

    Parameters
    ----------
    src : torch.Tensor
        The input tensor in degree representation.
    L : DegreeRange
        The degree range.
    dim : int
        The dimension along which to expand the tensor.

    Returns
    -------
    torch.Tensor
        The expanded tensor in order representation.
    """
    ind = order_batch(L, device=src.device)
    return src.index_select(dim, ind)

def reduce_order_to_degree(src: Tensor, L: DegreeRange, dim: int):
    """
    Reduce a tensor from order to degree representation.

    Parameters
    ----------
    src : torch.Tensor
        The input tensor in order representation.
    L : DegreeRange
        The degree range.
    dim : int
        The dimension along which to reduce the tensor.

    Returns
    -------
    torch.Tensor
        The reduced tensor in degree representation.
    """
    if dim < 0:
        dim = src.ndim + dim
    ind = order_ptr(L, dim=dim, device=src.device)
    return torch_geometric.utils.segment(src, ind)

@functools.lru_cache(maxsize=None)
def order_0_in(L: DegreeRange):
    L = check_degree_range(L)
    ls = torch.arange(L[0],L[1]+1)
    return ls+ls**2 - L[0]**2

def extract_in_degree(x: Tensor, L_x:DegreeRange, L:DegreeRange, dim=-2):
    L = check_degree_range(L)
    L_x = check_degree_range(L)
    return x.narrow(dim, L[0]**2-L_x[0]**2, num_order_between(*L))


def separate_invariant_equivariant(x:Tensor, dim=-2):
    return x.narrow_copy(dim=dim, start=0, length=1), \
           x.narrow_copy(dim=dim, start=0, length=x.shape[dim])

def concate_invariant_equivariant(invariant: Tensor, equivariant: Tensor, dim=-2):
    if invariant.ndim == equivariant.ndim -1:
        invariant = invariant.unsqueeze(dim)
    return torch.cat((invariant, equivariant), dim=dim)

