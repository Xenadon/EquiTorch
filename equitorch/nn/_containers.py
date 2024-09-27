import torch
import torch.nn as nn
from typing import Iterable, Dict, Any

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
        