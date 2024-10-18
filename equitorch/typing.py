from typing import Union, Tuple



DegreeRange = Union[Tuple[int,int], int]
r'''
The degree range.

Denoting the continuous range of degrees of a spherical tensor. 
It can be a tuple :math:`(l_{\text{min}}, l_{\text{max}})` 
or a single int :math:`l_{\text{max}}`. 
In Equitorch, a :obj:`DegreeRange` object is usually denoted by :math:`L`.
'''

__all__ = ['DegreeRange']