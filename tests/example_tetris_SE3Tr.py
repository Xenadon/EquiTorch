import sys
sys.path.append('..')

import time
from typing import Callable

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import SumAggregation
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms
from equitorch.nn._tfn import TFNBlock, SO2TFNBlock
from equitorch.nn._radial_basis import GaussianBasisExpansion
from equitorch.nn._cutoff import PolynomialCutoff
from equitorch.nn._activation import NormAct, ShiftedSoftPlus
from equitorch.nn._attention import AttentionalBlock, ScaledDotAttention
from equitorch.typing import DegreeRange
from equitorch.transforms import RadiusGraph, AddEdgeSphericalHarmonics, AddEdgeAlignWignerD

from e3nn import o3
from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork

class SE3TransformerBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int,
                 L_in: DegreeRange, 
                 L_out: DegreeRange,
                 channel_wise: bool = True,
                 act: Callable = None,
                ):
        super().__init__()

        self.attention_block = AttentionalBlock(
            L_in, L_out, in_channels,
            k_channels=out_channels//2,
            out_channels=out_channels//2*num_heads,
            attention_score_producer=ScaledDotAttention(L_in, L_in, in_channels, ))

class SE3Transformer(nn.Module):

    def __init__(self, hidden=4):

        super().__init__()
        self.hidden = hidden

        self.edge_embedding = GaussianBasisExpansion(0.1, 10, 0.7, 1.7)
        self.cutoff = PolynomialCutoff(1.5)
        self.layer1 = TFNBlock(in_channels=1, out_channels=hidden, 
                               L_in=0, L_edge=2, L_out=1, channel_wise=False, 
                               weight_producer=nn.Linear(10,2*hidden), act=NormAct(ShiftedSoftPlus(), 1))
        self.layer2 = TFNBlock(in_channels=hidden, out_channels=hidden, 
                               L_in=1, L_edge=2, L_out=1, channel_wise=True, 
                               weight_producer=nn.Linear(10,6*hidden), act=NormAct(ShiftedSoftPlus(), 1))
        self.layer3 = TFNBlock(in_channels=hidden, out_channels=hidden, 
                               L_in=1, L_edge=2, L_out=0, channel_wise=True, 
                               weight_producer=nn.Linear(10,2*hidden), act=nn.SiLU())
        self.pool = SumAggregation()
        self.output = nn.Sequential(nn.Linear(hidden, 8), nn.Softmax(dim=-1))

    def forward(self, x, edge_index, edge_sh, edge_len, batch):
        edge_emb = self.edge_embedding(edge_len)
        edge_weight = self.cutoff(edge_len)
        x = x.unsqueeze(-2)
        h = self.layer1(x, edge_index, edge_sh, edge_emb, edge_weight)
        h = self.layer2(h, edge_index, edge_sh, edge_emb, edge_weight)
        h = self.layer3(h, edge_index, edge_sh, edge_emb, edge_weight)
        h = self.pool(h[:,0,:], batch, dim=0)
        h = self.output(h)
        return h

class SO2TFN(nn.Module):

    def __init__(self, hidden=4):

        super().__init__()
        self.hidden = hidden

        self.edge_embedding = GaussianBasisExpansion(0.1, 10, 0.7, 1.7)
        self.cutoff = PolynomialCutoff(1.5)
        self.layer1 = SO2TFNBlock(in_channels=1, out_channels=hidden,
                               L_in=0, L_out=1, channel_wise=False, 
                               weight_producer=nn.Linear(10,2*hidden), act=NormAct(ShiftedSoftPlus(), 1))
        self.layer2 = SO2TFNBlock(in_channels=hidden, out_channels=hidden, 
                               L_in=1, L_out=1, channel_wise=True, 
                               weight_producer=nn.Linear(10,6*hidden), act=NormAct(ShiftedSoftPlus(), 1))
        self.layer3 = SO2TFNBlock(in_channels=hidden, out_channels=hidden, 
                               L_in=1, L_out=0, channel_wise=True, 
                               weight_producer=nn.Linear(10,2*hidden), act=nn.SiLU())
        self.pool = SumAggregation()
        self.output = nn.Sequential(nn.Linear(hidden, 8), nn.Softmax(dim=-1))

    def forward(self, x, edge_index, D1, edge_len, batch):
        DT1 = D1.transpose(-1,-2)
        D0 = DT0 = D1[:,:1,:1]
        x = x.unsqueeze(-2)
        edge_emb = self.edge_embedding(edge_len)
        edge_weight = self.cutoff(edge_len)
        h = self.layer1(x, edge_index, D0, DT1, edge_emb, edge_weight)
        h = self.layer2(h, edge_index, D1, DT1, edge_emb, edge_weight)
        h = self.layer3(h, edge_index, D1, DT0, edge_emb, edge_weight)
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

def make_batch(pos):
    # put in torch_geometric format
    transform = torch_geometric.transforms.Compose([
        RadiusGraph(r=1.5),
        AddEdgeSphericalHarmonics(L=2),
        AddEdgeAlignWignerD(L=1)
    ])
    dataset = [transform(Data(pos=pos, x=torch.ones(4, 1))) for pos in pos]
    return next(iter(DataLoader(dataset, batch_size=len(dataset))))

def Network() -> None:
    return SimpleNetwork(
        irreps_in="0e",
        irreps_out="8x0e",
        max_radius=1.5,
        num_neighbors=2.0,
        num_nodes=4.0,
    )

def main() -> None:
    torch.random.manual_seed(319)
    x, y = tetris()
    train_x, train_y = make_batch(x), y  # dont train on both chiral shapes

    x, y = tetris()
    test_x, test_y = make_batch(x), y

    hidden=16
    net = 'TFN'
    match net:
        case 'Network':
            f = Network()
        case 'TFN':    
            f = TFN(hidden)
        case 'SO2TFN':
            f = SO2TFN(hidden)
    # print("Built a model:")
    print(f)

    optim = torch.optim.Adam(f.parameters(), lr=1e-3)

    # == Training ==
    steps = 1000
    start_time = time.time()
    for step in range(1,steps+1):
        match net:
            case 'Network':
                pred = f(train_x).softmax(dim=-1)
            case 'TFN':    
                pred = f(train_x.x, train_x.edge_index, train_x.edge_sh, train_x.edge_vec.norm(dim=-1), train_x.batch)
            case 'SO2TFN':
                pred = f(train_x.x, train_x.edge_index, train_x.D, train_x.edge_vec.norm(dim=-1), train_x.batch)
        loss = torch.nn.functional.cross_entropy(pred, train_y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            match net:
                case 'Network':
                    pred = f(test_x).softmax(dim=-1)
                case 'TFN':    
                    pred = f(test_x.x, test_x.edge_index, test_x.edge_sh, test_x.edge_vec.norm(dim=-1), test_x.batch)
                case 'SO2TFN':
                    pred = f(test_x.x, test_x.edge_index, test_x.D, test_x.edge_vec.norm(dim=-1), test_x.batch)
            accuracy = pred.argmax(dim=-1).eq(test_y).double().mean(dim=0).item()
            print(f"epoch {step:5d} | loss {loss:<10.1f} | {100 * accuracy:5.1f}% accuracy")
    end_time = time.time()
    duration = end_time - start_time    
    print(f"The training took {duration:.2f} seconds to execute for {steps} steps")
    print()
    # # == Check equivariance ==
    # # Because the model outputs (psuedo)scalars, we can easily directly
    # # check its equivariance to the same data with new rotations:
    # print("Testing equivariance directly...")
    # rotated_x, _ = tetris()
    # rotated_x = make_batch(rotated_x)
    # pred_test = f(test_x.x, test_x.edge_index, test_x.edge_sh, test_x.edge_vec.norm(dim=-1), test_x.batch)
    # pred_rotated = f(rotated_x.x, rotated_x.edge_index, rotated_x.edge_sh, rotated_x.edge_vec.norm(dim=-1), rotated_x.batch)
    # # pred_test = f(test_x.x, test_x.edge_index, test_x.D, test_x.edge_vec.norm(dim=-1), test_x.batch)
    # # pred_rotated = f(rotated_x.x, rotated_x.edge_index, rotated_x.D, rotated_x.edge_vec.norm(dim=-1), rotated_x.batch)
    # error = pred_test - pred_rotated
    # print(f"Equivariance error = {error.abs().max().item():.1e}")



main()
