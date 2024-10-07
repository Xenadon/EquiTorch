from setuptools import setup

setup(
    name='equitorch',
    version='0.1',
    description='A package for constructing equivariant GNNs building upon pyg.',
    packages=['equitorch'],
    install_requires=[
        'pytorch>=2.2',
        'torch-geometric>=2.4',
        'e3nn>=0.5.1',
        'torch-scatter>=2.1.2'
    ],
    python_requires='>=3.12',
    author='Tong Wang',
    author_email='TongWang_2000@outlook.com',
    url='https://github.com/GTML-LAB/Equitorch/',
)
