import torch
import torch.nn as nn
from typing import Iterable, Dict, Any

class BranchedModuleList(nn.Module):
    """
    A module that applies a list of modules to the same input in parallel.

    :param modules: An iterable of PyTorch modules to be applied in parallel.
    :type modules: Iterable[nn.Module], optional

    :ivar modules: A ModuleList containing the parallel modules.
    :vartype modules: nn.ModuleList
    """
    def __init__(self, modules: Iterable[nn.Module] = None):
        super().__init__()
        self.modules = nn.ModuleList(modules)

    def forward(self, *args, **kwargs):
        """
        Applies each module in the ModuleList to the input arguments.

        :param args: Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
        :return: A list containing the outputs of each module in the ModuleList.
        :rtype: list
        """
        return [m(*args, **kwargs) for m in self.modules]

class BranchedModuleDict(nn.Module):
    """
    A module that applies a dictionary of modules to the same input in parallel.

    :param modules: A dictionary of PyTorch modules to be applied in parallel.
    :type modules: Dict[Any, nn.Module]

    :ivar modules: A ModuleDict containing the parallel modules.
    :vartype modules: nn.ModuleDict
    """
    def __init__(self, modules: Dict[Any,nn.Module]):
        super().__init__()
        self.modules = nn.ModuleDict(modules)

    def forward(self, *args, **kwargs):
        """
        Applies each module in the ModuleDict to the input arguments.

        :param args: Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
        :return: A dictionary containing the outputs of each module in the ModuleDict,
                 with the same keys as the input dictionary.
        :rtype: dict
        """
        return {k:m(*args, **kwargs) for k,m in self.modules.items()}
        