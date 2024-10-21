The tetris classification using SE(3)-Transformer with separable layernorm and S2 activation.
============================================================================================

.. note::
    For some backgrounds about the task and equivariant operation,
    you can first read `The tetris classification using TFN and SO2TFN`.

Here, we give an example of the tetris classification
using `SE(3)-Transformer <https://arxiv.org/abs/2006.10503>`_,
but replace the normalization and activation to the 
separable layer normalization and separable S2 activation
used in `EquiformerV2 <https://arxiv.org/abs/2306.12059>`_
to show the usage of these modules.

Though named with "Transformer", the SE(3)-Transformer in fact
is more like an attentional message passing neural network since
it does not have a "feed-forward" part like Equiformer/V2.
In the original version, the projector to produce Q,K,V are all
SO(3) equivariant linear operations. However, when implementing it,
we directly replace it using the equivalent :obj:`SO2Linear` 
operations. (In most of the cases we will use :obj:`SO2Linear`
rather than :obj:`SO3Linear` whenever possible.)

Besides, to show the usage of :obj:`Separable`, :obj:`EquivariantLayerNorm`
and :obj:`S2Act`, we also replace the normalization and activation
layer to separable layernorm and S2 activation.

Comparing to the training of the (SO2)TFN, we can also
observe a great acceleration in convergence, though
with more complicated operations.

.. code-block:: python

    import time

    import torch
    import torch.nn as nn
    import torch_geometric
    from torch_geometric.nn import SumAggregation, MessagePassing, Sequential
    from torch_geometric.data import Data, DataLoader
    import torch_geometric.transforms

    from equitorch.nn import (
        SO2Linear,
        GaussianBasisExpansion,
        PolynomialCutoff,
        S2Act,
        MultiheadAttentionBlock,
        SE3TrAttention,
        DegreeWiseLinear,
        EquivariantLayerNorm,
        Separable

    )
    from equitorch.utils.geometries import rot_on

    from equitorch.utils import num_degree_triplets, range_eq
    from equitorch.typing import DegreeRange
    from equitorch.transforms import RadiusGraph, AddEdgeAlignWignerD

    from e3nn import o3

    # Code borrowed and modified from https://github.com/e3nn/e3nn/blob/main/Example/tetris.py

    class SE3TransformerBlock(MessagePassing):

        def __init__(self,
                    in_channels: int,
                    out_channels: int,
                    L_in: DegreeRange, 
                    L_out: DegreeRange,
                    num_heads: int = 1,
                    k_channels: int = None,
                    L_k: DegreeRange = None,
                    ):
            super().__init__(node_dim=0)

            if k_channels is None:
                k_channels = in_channels
            if L_k is None:
                L_k = L_in
            
            self.num_weights_k = num_degree_triplets(L_in, L_k)
            self.num_weights_v = num_degree_triplets(L_in, L_out)
            self.att_msg = MultiheadAttentionBlock(
                num_heads=num_heads,
                attention_score_producer=SE3TrAttention(
                    L_in, L_in, in_channels, k_channels, num_heads,
                    nn.LazyLinear(self.num_weights_k*in_channels*num_heads*k_channels)
                ),
                v_producer=Sequential('x, edge_emb',[
                    (nn.LazyLinear(self.num_weights_v*in_channels*out_channels), 'edge_emb -> weight'),
                    (SO2Linear(L_in, L_out, in_channels, out_channels, True), 'x, weight -> x')
                ]) # The Sequential module provided by torch_geometric
            )
            self.self_interaction = DegreeWiseLinear(L_in, L_out, in_channels, out_channels)

            self.act = Separable(
                nn.Sequential(
                    nn.LayerNorm(out_channels),
                    nn.SiLU()
                ),
                nn.Sequential(
                    EquivariantLayerNorm(range_eq(L_out), out_channels),
                    S2Act(range_eq(L_out), nn.SiLU(), 8)
                )
            )

        def forward(self, x, edge_index,
                    D_in, DT_out, edge_emb, edge_weight = None):
            out = self.propagate(edge_index, x=x, 
                                edge_emb=edge_emb,
                                edge_weight=edge_weight,
                                D_in=D_in, DT_out=DT_out)
            out = out + self.self_interaction(x)
            return self.act(out)

        def message(self, x_j, x_i, edge_index,
                    edge_emb, edge_weight,
                    D_in, DT_out):
            x_i = rot_on(D_in, x_i)
            x_j = rot_on(D_in, x_j)
            out, _ = self.att_msg((x_i, x_j), x_j, edge_index[1], 
                                edge_emb=edge_emb)
            out = rot_on(DT_out, out)
            if edge_weight is not None:
                return edge_weight.view(-1,1,1) * out
            else:
                return out

    class SE3Transformer(nn.Module):

        def __init__(self, hidden=4, L=3, num_heads=1):

            super().__init__()
            self.hidden = hidden

            self.edge_embedding = GaussianBasisExpansion(0.1, 20, 0.7, 1.7)
            self.cutoff = PolynomialCutoff(1.5)

            self.layer1 = SE3TransformerBlock(in_channels=1, out_channels=hidden, k_channels=hidden//2,
                                            num_heads=1, L_in=0, L_out=L)
            self.layer2 = SE3TransformerBlock(in_channels=hidden, out_channels=hidden, k_channels=hidden//2,
                                            num_heads=num_heads, L_in=L, L_out=L)
            self.layer3 = SE3TransformerBlock(in_channels=hidden, out_channels=hidden, k_channels=hidden//2,
                                            num_heads=num_heads, L_in=L, L_out=L)
            self.layer4 = SE3TransformerBlock(in_channels=hidden, out_channels=hidden, k_channels=hidden//2,
                                            num_heads=num_heads, L_in=L, L_out=0)

            self.pool = SumAggregation()
            self.output = nn.Sequential(nn.Linear(hidden, 8), nn.Softmax(dim=-1))

        def forward(self, x, edge_index, D, edge_len, batch):
            DT = D.transpose(-1,-2)
            D0 = DT0 = D[:,:1,:1]
            x = x.unsqueeze(-2)
            edge_emb = self.edge_embedding(edge_len)
            edge_weight = self.cutoff(edge_len)
            h = self.layer1(x, edge_index, D0, DT, edge_emb, edge_weight)
            h = self.layer2(h, edge_index, D, DT, edge_emb, edge_weight)
            h = self.layer3(h, edge_index, D, DT, edge_emb, edge_weight)
            h = self.layer4(h, edge_index, D, DT0, edge_emb, edge_weight)
            h = self.pool(h[:,0,:], batch, dim=0)
            h = self.output(h)
            return h




    def tetris() -> None:
        pos = [
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
        ]
        pos = torch.tensor(pos, dtype=torch.get_default_dtype())
        labels = torch.arange(8, dtype=torch.long)

        # apply random rotation
        pos = torch.einsum("zij,zaj->zai", o3.rand_matrix(len(pos)), pos)

        return pos, labels

    L = 2
    hidden=16

    def make_batch(pos):
        # put in torch_geometric format
        transform = torch_geometric.transforms.Compose([
            RadiusGraph(r=1.5),
            AddEdgeAlignWignerD(L=L)
        ])
        dataset = [transform(Data(pos=pos, x=torch.ones(4, 1))) for pos in pos]
        return next(iter(DataLoader(dataset, batch_size=len(dataset))))

    def main() -> None:
        x, y = tetris()
        train_x, train_y = make_batch(x), y

        x, y = tetris()
        test_x, test_y = make_batch(x), y

        f = SE3Transformer(hidden, L=L, num_heads=4)
        print(f)

        optim = torch.optim.Adam(f.parameters(), lr=1e-3)

        # == Training ==
        steps = 50
        start_time = time.time()
        for step in range(1,steps+1):
            pred = f(train_x.x, train_x.edge_index, train_x.D, train_x.edge_vec.norm(dim=-1), train_x.batch)
            loss = torch.nn.functional.cross_entropy(pred, train_y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 10 == 0:
                pred = f(test_x.x, test_x.edge_index, test_x.D, test_x.edge_vec.norm(dim=-1), test_x.batch)
                accuracy = pred.argmax(dim=-1).eq(test_y).double().mean(dim=0).item()
                print(f"epoch {step:5d} | loss {loss:<10.1f} | {100 * accuracy:5.1f}% accuracy")
        end_time = time.time()
        duration = end_time - start_time    
        print(f"The training took {duration:.2f} seconds to execute for {steps} steps")
        print()

        # == Check equivariance ==
        # Since the S2Act used here is approximately equivariant
        # the output may not be exactly invariant  
        print("Testing equivariance directly...")
        rotated_x, _ = tetris()
        rotated_x = make_batch(rotated_x)
        pred_test = f(test_x.x, test_x.edge_index, test_x.D, test_x.edge_vec.norm(dim=-1), test_x.batch)
        pred_rotated = f(rotated_x.x, rotated_x.edge_index, rotated_x.D, rotated_x.edge_vec.norm(dim=-1), rotated_x.batch)
        error = pred_test - pred_rotated
        print(f"Equivariance error = {error.abs().max().item():.1e}")


    if __name__ == '__main__':
        main()
