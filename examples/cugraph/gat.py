import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import CuGraphGATConv, GATConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument("--mode", type=str, default="cugraph", choices=["pyg", "cugraph"])
args = parser.parse_args()
use_cugraph = args.mode == "cugraph"

device = 'cuda'

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

num_nodes = int(data.edge_index.max()) + 1

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, use_cugraph=True):
        super().__init__()
        Conv = CuGraphGATConv if use_cugraph else GATConv
        self.conv1 = Conv(in_channels, hidden_channels, heads)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = Conv(hidden_channels * heads, out_channels, heads=1,
                             concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads, use_cugraph=use_cugraph).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

if use_cugraph:
    edge_index = CuGraphGATConv.to_csc(data.edge_index, size=(num_nodes, num_nodes))
else:
    edge_index = data.edge_index

def train():
    model.train()
    optimizer.zero_grad()

    tic = time.time()
    out = model(data.x, edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    toc = time.time()
    return float(loss), toc - tic


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = final_test_acc = 0
t_model = 0
for epoch in range(1, args.epochs + 1):
    loss, t = train()
    t_model += t
    train_acc, val_acc, tmp_test_acc = test()

print(f"Time per epoch: {t_model / args.epochs:.4f}")
