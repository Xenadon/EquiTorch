conda create -n equiformer python=3.12
conda activate equiformer

CUDA=12.1

conda install numpy=1.26.4
conda install pytorch pytorch-cuda=$CUDA -c pytorch -c nvidia
conda install pyg torch-scatter torch-sparse torch-cluster torch-spline-conv -c pyg

pip install e3nn
pip install torch-harmonics