import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric import EdgeIndex
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import CuGraphGATConv
from torch_geometric.utils import add_self_loops, remove_self_loops

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError('CuGraphGATConv requires a CUDA-capable GPU')
device = torch.device('cuda')

init_wandb(
    name=f'CuGraphGAT-{args.dataset}', heads=args.heads,
    epochs=args.epochs, hidden_channels=args.hidden_channels, lr=args.lr,
    device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

# Preprocess the static graph once instead of rebuilding CSC and self-loops in
# every CuGraphGATConv call.
edge_index, _ = remove_self_loops(data.edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
edge_index = EdgeIndex(
    edge_index, sparse_size=(data.num_nodes, data.num_nodes))
data.edge_index = edge_index.sort_by('col')[0].fill_cache_()


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = CuGraphGATConv(
            in_channels, hidden_channels, heads, dropout=0.6,
            add_self_loops=False)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = CuGraphGATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False,
            dropout=0.6, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


times = []
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


model.eval()
with torch.no_grad():
    _, (attention_edge_index, alpha) = model.conv1(
        data.x, data.edge_index, return_attention_weights=True)
print(f'Attention edges: {attention_edge_index.size(1)}')
print(f'Attention weights shape: {tuple(alpha.shape)}')
