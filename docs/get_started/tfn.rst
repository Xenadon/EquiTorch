Our First Equivariant GNN - Tensor Field Network
================================================

.. note::

    This part assumes the reader have some preliminaries on conventional MPNNs (message passing neural networks) like GCN (graph convolutional networks).

    If you are fresh to these networks, you can refer to `one tutorial of pyg <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html>`_.

After having some basic ideas of spherical tensors, now we will build our first equivariant GNN, the *tensor field network*, from the aspect of MPNNs (message passing neural networks).

Message Passing Neural Networks
-------------------------------

Follows the definition used in pyg, a message passing layer that takes the node features :math:`\{\mathbf{x}_i\}` and edge features :math:`\{\mathbf{e}_{j,k}\}` can be formularized as

.. math::

    \mathbf{x}_i^{(k)}=\gamma^{(k)}\bigg(\mathbf{x}_i^{(k-1)},\bigoplus_{j\in\mathcal{N}(i)}\phi^{(k)}\big(\mathbf{x}_i^{(k-1)},\mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}^{(k-1)}\big)\bigg),

where :math:`\phi` is a message function that generate messages from both source node, target node and the edge; :math:`\bigoplus` is a differentiable, permutation-invariant function that aggregates messages from the neighbor :math:`\mathcal{N}(i)` and :math:`\gamma` is the update function that computes the new node embeddings considering the current embedding and the aggregated messages. 

For example, in the GCN