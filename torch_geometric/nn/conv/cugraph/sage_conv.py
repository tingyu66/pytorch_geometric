from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from torch_geometric import EdgeIndex

try:
    from cudnn.gnn import CscGraph, agg_simple
    HAS_CUDNN_GNN = True
except ImportError:
    HAS_CUDNN_GNN = False


class CuGraphSAGEConv(torch.nn.Module):  # pragma: no cover
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    :class:`CuGraphSAGEConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.SAGEConv` based on the cuDNN GNN API that
    fuses message passing computation for accelerated execution and lower
    memory footprint.

    Install :obj:`nvidia-cudnn-frontend[cutedsl]` to use this operator.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        if not HAS_CUDNN_GNN:
            raise ModuleNotFoundError(f"'{self.__class__.__name__}' requires "
                                      f"'nvidia-cudnn-frontend[cutedsl]'")

        if aggr not in ['mean', 'sum', 'min', 'max']:
            raise ValueError(f"Aggregation function must be either 'mean', "
                             f"'sum', 'min' or 'max' (got '{aggr}')")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if self.project:
            self.pre_lin = Linear(in_channels, in_channels, bias=True)

        if self.root_weight:
            self.lin = Linear(2 * in_channels, out_channels, bias=bias)
        else:
            self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.pre_lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeIndex,
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        if not isinstance(edge_index, EdgeIndex):
            raise ValueError(f"'edge_index' needs to be of type 'EdgeIndex' "
                             f"(got {type(edge_index)})")

        edge_index = edge_index.sort_by('col')[0]
        num_src_nodes = edge_index.get_sparse_size(0)
        (colptr, row), _ = edge_index.get_csc()

        if not row.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        graph = CscGraph(colptr, row, num_src_nodes)

        root = x[:graph.num_dst_nodes]

        if self.project:
            x = self.pre_lin(x).relu()

        out = agg_simple(
            graph,
            node_features=x,
            concat_features=root if self.root_weight else None,
            aggr=self.aggr,
        )

        out = self.lin(out)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
