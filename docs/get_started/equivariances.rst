Equivariances and Invariances
=============================

In this section, we would first give an introduction on the "`equivariances`" and "`Invariances`".

Intuitively, equivariances are the properties that "`when input gets transformed, the output gets transformed accordingly`", while the invariances means that "`when input gets transformed, the output keeps unchanged`".

Mathematically, the equivariances and invariances can be defined via group representations.

Suppose we are given a group :math:`G`, such as 3D-rotations :math:`\text{SO(3)}`. Beyond the original transformation on :math:`\mathbb{R}^3` by its definition, we may also **represent** the group elements as transformation on other spaces, while preserving the multiplication relation of original group elements. If we are able to do that, we are said to found a **representation** of that group.

Formally, a **representation** of a group :math:`G` on a linear space :math:`X` is a mapping :math:`\rho_X:G\to \mathrm{GL}(X)`, such that for any two members :math:`a,b` of :math:`G`, it holds that

.. math::
    \rho_X(a)\cdot \rho_X(b)=\rho_X(a\cdot b),

where :math:`\mathrm{GL}(X)` is the general linear group on :math:`X`, that is, all invertible transformations on it.

If we have two representations :math:`\rho_X,\rho_Y` for a group :math:`G` on linear spaces :math:`X` and :math:`Y`, respectively, a function :math:`f:X\to Y` is said to be **equivariant** if 

.. math::
    f(\rho_X(g) x)=\rho_Y(g)f(x)

holds for any :math:`g\in G` and :math:`x\in X`. In particular, if :math:`\rho_Y(g)\equiv I_Y`, that is the identity on :math:`Y`, the equivariance condition will become

.. math::
    f(\rho_X(g) x)=f(x)

for any :math:`g\in G` and :math:`x\in X`. In this special case, we call the function :math:`f` to be **invariant**.

Equivariances and invariances are crucial properties in many machine learning tasks. 

For example, in object recognition tasks, input images can be viewed as elements in the space :math:`X` of signals :math:`h(\mathbf{x})` over :math:`\mathbb{R}^2`. A translation :math:`\mathbf{t}\in\mathbb{R}^2` will act on the input image by

.. math::
    \rho_X(\mathbf{t}):h(\mathbf{x})\mapsto h(\mathbf{x}-\mathbf{t}),

whereas the  center :math:`\mathbf{c}\in Y=\mathbb{R}^2` of the bounding box is translated by

.. math::
    \rho_Y(\mathbf{t}):\mathbf{c}\mapsto \mathbf{c}+\mathbf{t}.

A neural network :math:`F_c` to predict the center should statisfy

.. math::
    F_c[h(\mathbf{x}-\mathbf{t})]=F_c[h(\mathbf{x})]+\mathbf{t},

meaning that the translated image will lead to a translated bounding box, which is equivariant. 

On the other hand, if we are using another network :math:`F_s` to predict the size (width and height) :math:`\mathbf{s}\in Y'=\mathbb{R}_+^2` of the bounding box, the translation should not affect the result, that is,

.. math::
    F_s[h(\mathbf{x}-\mathbf{t})] = F_s[h(\mathbf{x})],

which is invariant.

Convolutional neural networks (CNNs) are naturally aligned with these properties: convolution layers are inherently translation-equivariant, while global pooling layers introduce translation invariance.

Similarly, for graph-related tasks, to permute node indicies by a permutation :math:`\mathbf{P}` will not affect the result. A graph can be denoted as a pair :math:`(\mathbf{X}, \mathbf{A})` where :math:`\mathbf{X}` is the node features and :math:`\mathbf{A}` is the adjacency matrix and an index permutation :math:`\mathbf{P}` will act on the graph by 

.. math::
    \rho_X(\mathbf{P}): (\mathbf{X}, \mathbf{A})\mapsto (\mathbf{PX}, \mathbf{PAP}^\top),

Node classification :math:`\mathbf{y}=f(\mathbf{X}, \mathbf{A})` should be permutation-equivariant, in which the permuted input should lead to the permuted output :math:`\mathbf{Py}`. On the other hand, graph classification :math:`\mathbf{y}=f(\mathbf{X}, \mathbf{A})` should be permutation-invariant, that is, the permuted input should still give the same label :math:`y`. In GNNs, The massage-passing layers are permutation-equivariant, and the global pooling layers are permutation-invariant.

In many scientific tasks, we hope to model the rotation-equivariances or invariances