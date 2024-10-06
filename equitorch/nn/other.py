import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Iterable, Dict, Any

from ..utils.hooks import get_kwargs_filter_hook
from ..utils.indices import concate_invariant_equivariant, separate_invariant_equivariant

from ..typing import DegreeRange

class BranchedModuleList(nn.Module):
    """
    A module that applies a list of modules to the same input in parallel.

    Parameters
    ----------
    modules : Iterable[nn.Module], optional
        An iterable of PyTorch modules to be applied in parallel.
    """
    def __init__(self, modules: Iterable[nn.Module] = None):
        super().__init__()
        self.modules = nn.ModuleList(modules)

    def forward(self, *args, **kwargs):
        """
        Applies each module in the ModuleList to the input arguments.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        list
            A list containing the outputs of each module in the ModuleList.
        """
        return [m(*args, **kwargs) for m in self.modules]

class BranchedModuleDict(nn.Module):
    """
    A module that applies a dictionary of modules to the same input in parallel.

    Parameters
    ----------
    modules : Dict[Any, nn.Module]
        A dictionary of PyTorch modules to be applied in parallel.

    Attributes
    ----------
    modules : nn.ModuleDict
        A ModuleDict containing the parallel modules.
    """
    def __init__(self, modules: Dict[Any,nn.Module]):
        super().__init__()
        self.modules = nn.ModuleDict(modules)

    def forward(self, *args, **kwargs):
        """
        Applies each module in the ModuleDict to the input arguments.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        dict
            A dictionary containing the outputs of each module in the ModuleDict,
            with the same keys as the input dictionary.
        """
        return {k:m(*args, **kwargs) for k,m in self.modules.items()}
        
class Separable(nn.Module):
    """
    Separable operation for invariant and equivariant features.

    Parameters
    ----------
    op_inv: Callable
        operation for invariant components.
    op_eqv: Callable
        operation for equivariant components.
    cat_after: bool, optional
        Whether to concatenate the invariant and equivariant outputs.
        Default is :obj:`True`.

    Example
    -------
    >>> act = Separable(
    >>>     torch.nn.SiLU(),
    >>>     S2Act(8, nn.SiLU(), range_eq(L_out)）
    >>> )
    """
    def __init__(self, op_inv: Callable, op_eqv: Callable,
                 cat_after: bool = True):
        super().__init__()
        self.op_inv = op_inv
        self.op_eqv = op_eqv
        self.cat_after = cat_after

        if isinstance(self.op_inv, nn.Module):
            self.op_inv.register_forward_pre_hook(
                get_kwargs_filter_hook(self.op_inv), with_kwargs=True)

        if isinstance(self.op_eqv, nn.Module):
            self.op_eqv.register_forward_pre_hook(
                get_kwargs_filter_hook(self.op_eqv), with_kwargs=True)


    def forward(self, x: Tensor, dim: int=-2, **kwargs):
        """
        Parameters
        ----------
        x: Tensor
            The input tensor. Must have invariant (:math:`l=0`) components but 
            can have no equivariant (:math:`l>0`) components.
        dim: int, optional
            The dimension of spherical orders. Default is :obj:`-2`.
        **kwargs:
            Any keyword arguments that will be passed to
            :obj:`op_inv` and :obj:`op_eqv` 
        """
        xi, xe = separate_invariant_equivariant(x,dim)
        xi = self.op_inv(xi.squeeze(dim), **kwargs).unsqueeze(dim)
        if xe.numel() > 0:
            xe = self.op_eqv(xe, **kwargs)
        if self.cat_after:
            return concate_invariant_equivariant(xi, xe, dim)
        else:
            return xi, xe