import argparse
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.datasets import Entities
from torch_geometric.nn import CuGraphRGCNConv, FastRGCNConv as RGCNConv
from torch_geometric.loader import NeighborLoader

assert torch_geometric.typing.WITH_PYG_LIB

parser = argparse.ArgumentParser(
    description="RGCN for entity classification with neighbor sampling."
)
parser.add_argument("--mode", type=str, default="cugraph", choices=["pyg", "cugraph"])
parser.add_argument('--dataset', type=str, choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
args = parser.parse_args()
use_cugraph = args.mode == "cugraph"

device = "cuda:0"
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'Entities')
dataset = Entities(path, args.dataset)
data = dataset[0].to(device, 'edge_index', 'edge_type', 'train_y', 'test_y')

num_bases = 10

class RGCN(torch.nn.Module):
    def __init__(self, use_cugraph=True):
        super().__init__()
        self.emb = nn.Embedding(dataset.num_nodes, 16)
        Conv = CuGraphRGCNConv if use_cugraph else RGCNConv
        self.conv1 = Conv(16, 16, dataset.num_relations, num_bases=num_bases)
        self.conv2 = Conv(16, dataset.num_classes, dataset.num_relations, num_bases=num_bases)

    def forward(self, n_id, edge_index, edge_type):
        """ edge_index, edge_type needed to be in csc representation when `use_cugraph=True`."""
        x = self.emb(n_id)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x

loader = NeighborLoader(data, num_neighbors=[30,30], batch_size=1024, input_nodes=data.train_idx)
test_loader = NeighborLoader(data, num_neighbors=[-1,-1], batch_size=1024, input_nodes=data.test_idx)
model = RGCN(use_cugraph=use_cugraph).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

@torch.no_grad()
def test():
    model.eval()
    total_correct = total_examples = 0
    for batch in test_loader:
        if use_cugraph:
            edge_index, edge_type = CuGraphRGCNConv.to_csc(batch.edge_index, size=(batch.num_nodes, batch.num_nodes), edge_attr=batch.edge_type)
        else:
            edge_index, edge_type = batch.edge_index, batch.edge_type
        n_id = batch.n_id.to(device)
        y_hat = model(n_id, edge_index, edge_type)[:batch.batch_size]
        y = batch.test_y[total_examples:total_examples+batch.batch_size]
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
    return total_correct / total_examples

def train():
    model.train()
    t_model = 0
    n_epoch = 100
    for epoch in range(n_epoch):
        total_loss = total_correct = total_examples = 0
        for batch in loader:
            optimizer.zero_grad()

            if use_cugraph:
                edge_index, edge_type = CuGraphRGCNConv.to_csc(batch.edge_index, size=(batch.num_nodes, batch.num_nodes), edge_attr=batch.edge_type)
            else:
                edge_index, edge_type = batch.edge_index, batch.edge_type
            n_id = batch.n_id.to(device)

            tic = time.time()

            y_hat = model(n_id, edge_index, edge_type)[:batch.batch_size]
            y = batch.train_y[total_examples:total_examples+batch.batch_size]
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

            t_model += time.time() - tic

            total_loss += float(loss) * batch.batch_size
            total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            total_examples += batch.batch_size

        loss = total_loss / total_examples
        train_acc = total_correct / total_examples
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f} ')
    test_acc = test()
    print(f'Model time per epoch (s): {t_model / n_epoch:.4f}')
    print(f'Final Test accuracy: {test_acc:.4f}')

train()
