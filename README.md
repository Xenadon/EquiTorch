# Equitorch

![Equitorch_logo](./img/logo_wide.png)

*Equitorch* is a modularized package that can be used to flexibly constructing equivariant GNNs based on Pytorch-Geometric.

**[Github Pages](https://github.com/GTML-LAB/Equitorch/tree/main)**

**[Documentation](https://equitorch.readthedocs.io/en/latest/index.html)**

> This package is still under development. 
> We are actively adding more operations, documentations and tutorials.

In this package, we implemented many basic operators that may need in equivariant neural networks, currently including:

- Modules:
    - Equivariant linear transformations
    - (Weighted) Tensor Product between spherical tensors 
    - Equivariant activation
    - Equivariant normalization
    - Attention operations
    - Basis expansion (radial, angular and spherical)
    - Cutoff operations
- Mathematical functions:
    - Operations related to spherical tensors, spherical harmonics and Wigner D matrices
    - Spherical harmonic transform and inverse spherical harmonic transform
- Transforms
- Utility functions

### Installation

This package is based on [Pytorch](https://pytorch.org/)(>=2.2), [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)(>=2.4). Please make sure you have already installed the version that fit your device. (It is temporarily recommended to use `pip` to install the Pytorch-Geometric.)

With these packages installed, you can install *Equitorch* by

```bash
pip install git+https://github.com/GTML-LAB/Equitorch.git
```