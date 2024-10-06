.. equitorch documentation master file, created by
   sphinx-quickstart on Tue Sep 17 14:07:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Equitorch documentation
=======================

This is **Equitorch**, a modularized package for flexibily constructing equivariant (and invariant) GNNs built-upon pyg_.


.. .. toctree:
..    :maxdepth: 2
..    :caption: Get-started

   .. get_started/symmetries_equivariances
   .. get_started/representations_spherical_harmonics
   .. get_started/tensor_products



.. toctree:
..    :maxdepth: 2
..    :caption: Tutorials

.. toctree::
   :maxdepth: 1
   :caption: Examples

   /examples/teris_TFN.rst 
   /examples/teris_SE3Tr.rst 

.. toctree::
   :maxdepth: 1
   :caption: API 


   .. equivariance_introduction.rst

   /modules/typing.rst
   /modules/nn.rst
   /modules/math.rst
   /modules/utils.rst
   /modules/transforms.rst



.. _pyg: https://pytorch-geometric.readthedocs.io/en/latest/index.html