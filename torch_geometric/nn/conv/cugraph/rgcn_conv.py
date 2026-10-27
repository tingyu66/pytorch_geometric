from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric import EdgeIndex
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Size

try:
    from cudnn.gnn import CscGraph, agg_hg_basis_post
    HAS_CUDNN_GNN = True
except ImportError:
    HAS_CUDNN_GNN = False


class CuGraphRGCNConv(torch.nn.Module):  # pragma: no cover
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.

    :class:`CuGraphRGCNConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.RGCNConv` that uses the cuDNN GNN API to
    fuse relation-wise aggregation and optional basis combination.

    Install :obj:`nvidia-cudnn-frontend` with a cuDNN 9.29 or newer backend to
    use this operator. Only FP32 node features are supported.
    """
    def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                 num_bases: Optional[int] = None, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True):
        super().__init__()

        if not HAS_CUDNN_GNN:
            raise ModuleNotFoundError(
                f"'{self.__class__.__name__}' requires "
                f"'nvidia-cudnn-frontend' and cuDNN >= 9.29")

        if aggr not in ['sum', 'add', 'mean']:
            raise ValueError(f"Aggregation function must be either 'mean' "
                             f"or 'sum' (got '{aggr}')")
        if num_bases is not None and num_bases <= 0:
            raise ValueError("'num_bases' must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.aggr = aggr
        self.root_weight = root_weight

        dim_root_weight = 1 if root_weight else 0

        if num_bases is not None:
            self.weight = Parameter(
                torch.empty(num_bases + dim_root_weight, in_channels,
                            out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))
        else:
            self.weight = Parameter(
                torch.empty(num_relations + dim_root_weight, in_channels,
                            out_channels))
            self.register_parameter('comp', None)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        end = -1 if self.root_weight else None
        glorot(self.weight[:end])
        glorot(self.comp)
        if self.root_weight:
            glorot(self.weight[-1])
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        size: Size = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            edge_type (torch.Tensor): The edge type.
            size ((int, int), optional): The number of source and destination
                nodes. When omitted, the size is inferred from
                :obj:`edge_index` metadata or :obj:`x`.
        """
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(f"'{self.__class__.__name__}' does not support "
                               f"deterministic backward")
        if x.dim() != 2:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"static graphs")
        if not x.is_cuda or not edge_index.is_cuda or not edge_type.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")
        if x.dtype != torch.float32:
            raise TypeError(f"'{self.__class__.__name__}' requires FP32 node "
                            f"features (got {x.dtype})")

        if size is None:
            sparse_size: Tuple[Optional[int], Optional[int]] = (None, None)
            if isinstance(edge_index, EdgeIndex):
                sparse_size = edge_index.sparse_size()
            num_src_nodes = sparse_size[0] or x.size(0)
            num_dst_nodes = sparse_size[1] or x.size(0)
            size = (num_src_nodes, num_dst_nodes)

        if size[0] != x.size(0):
            raise ValueError(f"Expected {size[0]} source node features (got "
                             f"{x.size(0)})")
        if size[1] is None:
            raise ValueError("The number of destination nodes must be known")

        edge_index = EdgeIndex(edge_index, sparse_size=size)
        edge_index, perm = edge_index.sort_by('col', stable=True)
        (colptr, row), _ = edge_index.get_csc()
        if edge_type.dim() != 1 or edge_type.numel() != row.numel():
            raise ValueError(f"'edge_type' must contain one value per edge "
                             f"(expected {row.numel()}, got "
                             f"{edge_type.numel()})")
        if perm is not None:
            edge_type = edge_type[perm]
        edge_type = edge_type.to(dtype=torch.int32)

        graph = CscGraph(colptr, row, num_src_nodes=size[0])

        if row.numel() == 0 and size[1] > 0:
            num_groups = (self.num_relations if self.comp is None else
                          self.num_bases)
            assert num_groups is not None
            out = x.new_zeros((size[1], num_groups * self.in_channels))
            out = out + x.sum() * 0
            if self.comp is not None:
                out = out + self.comp.sum() * 0
            if self.root_weight:
                if size[1] > size[0]:
                    raise ValueError("Root weights require destination nodes "
                                     "to be a prefix of source nodes")
                out = torch.cat([out, x[:size[1]]], dim=-1)
        else:
            out = agg_hg_basis_post(
                graph,
                node_features=x,
                edge_types=edge_type,
                num_edge_types=self.num_relations,
                weights_comb=self.comp,
                concat_own=self.root_weight,
                norm_by_edge_type=self.aggr == 'mean',
            )

        out = out @ self.weight.view(-1, self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
