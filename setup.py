from setuptools import setup, find_packages
import equitorch

setup(
    name='equitorch',
    version=equitorch.__version__,
    description='A package for constructing equivariant GNNs building upon pyg.',
    packages=find_packages(),
    package_data={'': ['*.pt']},
    install_requires=[
        'e3nn',
        'torch-cluster'
    ],
    python_requires='>=3.12',
    author='Tong Wang',
    author_email='TongWang_2000@outlook.com',
    url='https://github.com/GTML-LAB/Equitorch/',
)
