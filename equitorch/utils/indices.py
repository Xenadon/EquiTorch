from typing import Tuple, List, Optional, Callable
import torch
from torch.nn.functional import pad
from torch import Tensor
import functools
import torch_geometric
from torch_geometric.utils import scatter, segment
import torch_geometric.utils

from ..typing import DegreeRange

def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr

def expand_right(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(-1)
    return ptr

@functools.lru_cache(maxsize=None)
def check_degree_range(L: DegreeRange) -> Tuple[int,int]:
    """
    Check and normalize the degree range.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
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

def degrees_in_range(L: DegreeRange):
    """
    Returns :obj:`range(L[0], L[1]+1)`.
    """
    L = check_degree_range(L)
    return range(L[0], L[1]+1)

@functools.lru_cache(maxsize=None)
def order_batch(L: DegreeRange, device=None):
    """
    Generate a tensor of degree indices for each order.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    device : torch.device, optional
        The device to place the resulting tensor on.

    Returns
    -------
    :obj:`~torch.Tensor`
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


@functools.lru_cache(maxsize=None)
def order_ptr(L: DegreeRange, dim:int=0, device=None):
    """
    Generate a tensor of pointers to the start of each degree's orders, including the next of the final elements.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    dim : int, optional
        The dimension along which to expand the result. Default is 0.
    device : torch.device, optional
        The device to place the resulting tensor on.

    Returns
    -------
    :obj:`~torch.Tensor`
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

def orders_in_degree_range(L: DegreeRange, zero_based:bool=True, device=None):
    r"""
    Generate a tensor that specifying the order :math:`m` in a feature of 
    degree range :math:`L`.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    zero_based : bool, optional
        Whether the :math:`m` is in :math:`0,\dots,2l_\text{max}+1` or 
        :math:`-l_\text{max},\dots,l_\text{max}`. Default is :obj:`True`.
    device : torch.device, optional
        The device to place the resulting tensor on.

    Returns
    -------
    :obj:`~torch.Tensor`
        A tensor denoting the degrees.

    Example
    -------
    >>> L = (1, 2)
    >>> result = orders_in_degree_range(L, True)
    >>> print(result)
    tensor([1, 2, 3, 0, 1, 2, 3, 4])
    >>> result = orders_in_degree_range(L, False)
    >>> print(result)
    tensor([-1, 0, 1, -2, -1, 0, 1, 2])
    """
    return torch.tensor([
        m+zero_based*(L[1]) for l in degrees_in_range(L) for m in orders_in_degree(l)
    ], device=device)

def orders_in_degree(l:int):
    """
    returns :obj:`range(-l, l+1)`
    """
    return range(-l, l+1)

@functools.lru_cache(maxsize=None)
def list_degrees(L: DegreeRange, L1: DegreeRange, L2: DegreeRange = None, cond: Optional[Callable[[int, int, int], bool]] =None):
    """
    Generate a list of valid degree triplets (l, l1, l2). (The triplet satisfies triangular inequality)

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The range for the total angular momentum l.
    L1 : :obj:`~equitorch.typing.DegreeRange`
        The range for the first angular momentum l1.
    L2 : :obj:`~equitorch.typing.DegreeRange`
        The range for the second angular momentum l2. default: None, represents the range of all possible l2's
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
    L2 = check_degree_range(L2) if L2 is not None \
        else (0, L[1] + L1[1])
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

@functools.lru_cache(maxsize=None)
def num_degree_triplets(L: DegreeRange, 
                        L1: DegreeRange, 
                        L2: Optional[DegreeRange] = None, 
                        cond: Optional[Callable[[int, int, int], bool]] = None):
    r"""
    Generate the number of valid degree triplets (l, l1, l2). (The triplet satisfies triangular inequality)

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The range for the total angular momentum l.
    L1 : :obj:`~equitorch.typing.DegreeRange`
        The range for the first angular momentum l1.
    L2 : :obj:`~equitorch.typing.DegreeRange`, optional
        The range for the second angular momentum l2
        Default is :obj:`None`. 
        If not provided, it will be :math:`[0,\max(L1)+\max(L2)]`
    cond : Callable[[int, int, int], bool], optional
        An optional condition function to further filter the triplets.

    Returns
    -------
    int
        The number of valid degree triplets.

    Example
    -------
    >>> print(num_degree_triplets(1, (1,2), (2,3)))
    4
    """
    return len(list_degrees(
        check_degree_range(L),
        check_degree_range(L1),
        check_degree_range(L2), cond))

def num_orders_in(L: DegreeRange):
    """
    Calculate the number of spherical orders in a degree range.
    """
    L = check_degree_range(L)
    return num_orders_between(*L)


def num_orders_between(l_min: int, l_max: int):
    """
    Calculate the number of orders between two degrees.

    Parameters
    ----------
    l_min : int
        The minimum degree.
    l_max : int
        The maximum degree.

    Returns
    -------
    int
        The total number of orders between l_min and l_max (inclusive).
    """
    return (l_max+1) ** 2 - l_min ** 2

def num_orders_of_degree(l: int):
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
    src : :obj:`~torch.Tensor`
        The input tensor in degree representation.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    dim : int
        The dimension along which to expand the tensor.

    Returns
    -------
    :obj:`~torch.Tensor`
        The expanded tensor in order representation.
    """
    ind = order_batch(L, device=src.device)
    return src.index_select(dim, ind)

def reduce_order_to_degree(src: Tensor, L: DegreeRange, dim: int):
    """
    Reduce a tensor from order to degree representation.

    Parameters
    ----------
    src : :obj:`~torch.Tensor`
        The input tensor in order representation.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range.
    dim : int
        The dimension along which to reduce the tensor.

    Returns
    -------
    :obj:`~torch.Tensor`
        The reduced tensor in degree representation.
    """
    if dim < 0:
        dim = src.ndim + dim
    ind = order_ptr(L, dim=dim, device=src.device)
    return torch_geometric.utils.segment(src, ind)


def extract_in_degree(x: Tensor, L_x:DegreeRange, L:DegreeRange, dim=-2):
    """
    Extracts a subset of degrees from the input tensor `x` along the specified dimension.

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        The input tensor containing data across degrees.
    L_x : :obj:`~equitorch.typing.DegreeRange`
        The degree range of the input tensor `x`.
    L : :obj:`~equitorch.typing.DegreeRange`
        The degree range to extract from `x`. 
        This must be within `L_x` (i.e., `L_x[0] <= L[0] <= L[1] <= L_x[1]`).
    dim : int, optional
        The dimension along which the degrees are stored in `x`. Default is -2.

    Returns
    -------
    :obj:`~torch.Tensor`
        A tensor containing only the degrees specified by `L`.

    Example
    -------
    >>> x = torch.randn(3, 25, 2)
    >>> result = extract_in_degree(x, (0, 4), (2, 3))
    >>> print(result.shape)
    torch.Size([3, 12, 2])
    ```
    """
    L = check_degree_range(L)
    L_x = check_degree_range(L_x)
    return x.narrow(dim, L[0]**2-L_x[0]**2, num_orders_between(*L))

def pad_to_degree(x: Tensor, L_x:DegreeRange, L:DegreeRange, dim=-2):
    """
    Pads the input tensor `x` to match the target degree range `L` by adding zeros.

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        The input tensor containing data across degrees.
    L_x : Tuple[int, int]
        The degree range `(min_degree, max_degree)` of the input tensor `x`.
    L : Tuple[int, int]
        The target degree range `(min_degree, max_degree)` to pad `x` to. 
        `L_x` must be within `L` (i.e., :obj:`L[0] <= L_x[0] <= L_x[1] <= L[1]`).
    dim : int, optional
        The dimension along which the degrees are stored in `x`. Default is -2

    Returns
    -------
    :obj:`~torch.Tensor`
        A tensor padded with zeros to match the degree range `L`.

    Example
    -------
    >>> x = torch.randn(3, 12, 2)
    >>> result = pad_to_degree(x, (2, 3), (0, 4))
    >>> print(result.shape)
    torch.Size([3, 25, 2])
    """
    L = check_degree_range(L)
    L_x = check_degree_range(L_x)
    pad_head = L_x[0]**2-L_x[0]**2
    pad_tail = (L[1]+1)**2 - (L_x[1]+1)**2
    if dim >= 0:
        dim = dim - x.ndim
    dim = -dim - 1
    p = (0,0)*dim + (pad_head, pad_tail)
    return pad(x, p)

def range_eq(L: DegreeRange):
    """
    Adjusts the input degree range `L` to ensure that it only includes 
    degrees greater than or equal to 1 for 
    strictly equivariant components.

    Parameters
    ----------
    L : :obj:`~equitorch.typing.DegreeRange`
        The input degree range.

    Returns
    -------
    :obj:`~equitorch.typing.DegreeRange`

    Examples
    --------
    >>> print(range_eq(4))
    (1, 4)
    >>> print(range_eq((2, 5)))
    (2, 5)
    """
    return (1, L) if isinstance(L, int) else (max(1,L[0]), L[1])

def separate_invariant_equivariant(x:Tensor, dim=-2):
    """Separate the invariant and equivariant components of the input.

    Warning
    -------
    It will produce meaningless results if the passed tensor
    do not have invariant components.

    Parameters
    ----------
    x : :obj:`~torch.Tensor`
        The input tensor.
    dim : int, optional
        The spherical dimension. Default is -2

    Returns
    -------
    (:obj:`~torch.Tensor`, :obj:`~torch.Tensor`)
        The invariant component and strictly equivariant component
        of the input tensor. (The equivariant component may be length-0
        in dim if the input is length-1 in dim)
    """
    return x.narrow(dim=dim, start=0, length=1), \
           x.narrow(dim=dim, start=1, length=x.shape[dim]-1)

def concate_invariant_equivariant(invariant: Tensor, equivariant: Tensor, dim=-2):
    """Concatenate invariant and strictly equivariant components.

    Warning
    -------
    It will produce meaningless results if the passed tensor
    is not with degrees 0 and (1,l_max).


    Parameters
    ----------
    invariant : :obj:`~torch.Tensor`
        The invariant components. 
        It may not contains the spherical dimension and thus 
        have one less dimension than the equivariant part. 
    equivariant : :obj:`~torch.Tensor`
        The equivariant components.
    dim: int, optional
        The spherical dimension. Default is -2

    Returns
    -------
    :obj:`~torch.Tensor`
        The concatenated tensor.
    """
    if invariant.ndim == equivariant.ndim -1:
        invariant = invariant.unsqueeze(dim)
    return torch.cat((invariant, equivariant), dim=dim)

def reduce(x: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
            dim: int = -2, reduce: str = 'sum', deterministic=False) -> Tensor:

    if ptr is not None:
        if index is None or deterministic:
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment(x, ptr, reduce=reduce)

    if index is None:
        raise RuntimeError("Aggregation requires 'index' to be specified")

    return scatter(x, index, dim, dim_size, reduce)