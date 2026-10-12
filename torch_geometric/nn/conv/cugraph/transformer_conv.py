from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import PairTensor

try:
    from cudnn.gnn import CscGraph, mha_simple
    HAS_CUDNN_GNN = True
except ImportError:
    HAS_CUDNN_GNN = False


class CuGraphTransformerConv(torch.nn.Module):  # pragma: no cover
    r"""A CUDA-optimized version of
    :class:`~torch_geometric.nn.conv.TransformerConv` backed by cuDNN's
    simple multi-head attention operation.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method. A
            tuple corresponds to source and target feature dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, combines aggregation and skip
            information through a learned gate. (default: :obj:`False`)
        dropout (float, optional): Dropout probability of normalized attention
            coefficients. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality.
            (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            additive biases. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, transformed root
            node features are not added to the output and :obj:`beta` is
            disabled. (default: :obj:`True`)

    Install :obj:`nvidia-cudnn-frontend` with a cuDNN 9.28 or newer backend to
    use this operator.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()

        if not HAS_CUDNN_GNN:
            raise ModuleNotFoundError(
                f"'{self.__class__.__name__}' requires "
                f"'nvidia-cudnn-frontend' and cuDNN >= 9.28")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        channels = ((in_channels, in_channels)
                    if isinstance(in_channels, int) else in_channels)
        self.lin_key = Linear(channels[0], heads * out_channels, bias=bias)
        self.lin_query = Linear(channels[1], heads * out_channels, bias=bias)
        self.lin_value = Linear(channels[0], heads * out_channels, bias=bias)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        total_out_channels = out_channels * (heads if concat else 1)
        self.lin_skip = Linear(channels[1], total_out_channels, bias=bias)
        if self.beta:
            self.lin_beta = Linear(3 * total_out_channels, 1, bias=False)
        else:
            self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.lin_beta is not None:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        x_src, x_dst = (x, x) if isinstance(x, Tensor) else x
        if x_src.dim() != 2 or x_dst.dim() != 2:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"static graphs")
        if not x_src.is_cuda or not x_dst.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        size = (x_src.size(0), x_dst.size(0))
        output_edge_index = edge_index
        edge_index = EdgeIndex(edge_index, sparse_size=size)
        edge_index, perm = edge_index.sort_by('col', stable=True)
        (colptr, row), _ = edge_index.get_csc()
        if perm is not None and perm.dtype != colptr.dtype:
            perm = perm.to(dtype=colptr.dtype)

        graph = CscGraph(
            colptr,
            row,
            num_src_nodes=size[0],
            map_csc_to_coo=perm,
        )
        deterministic = torch.are_deterministic_algorithms_enabled()
        if deterministic:
            graph = graph.with_reverse_csc()

        key = self.lin_key(x_src)
        query = self.lin_query(x_dst)
        value = self.lin_value(x_src)

        edge_features = None
        if self.lin_edge is not None:
            if edge_attr is None:
                raise ValueError("Edge features are required when "
                                 "'edge_dim' is specified")
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_features = self.lin_edge(edge_attr)
        elif edge_attr is not None:
            raise ValueError("Cannot use 'edge_attr' without specifying "
                             "'edge_dim'")

        dropout_mask = None
        if self.training and self.dropout > 0.0:
            dropout_mask = F.dropout(
                torch.ones(
                    (row.numel(), self.heads),
                    device=x_src.device,
                    dtype=torch.float32,
                ),
                p=self.dropout,
                training=True,
            ).t()

        return_attention = isinstance(return_attention_weights, bool)
        if row.numel() == 0:
            dim = self.out_channels * (self.heads if self.concat else 1)
            out = query.new_zeros((size[1], dim))
            out = out + (key.sum() + query.sum() + value.sum()) * 0
            if edge_features is not None:
                out = out + edge_features.sum() * 0
            attention = torch.empty(
                (0, self.heads),
                device=x_src.device,
                dtype=x_src.dtype,
            )
        else:
            result = mha_simple(
                graph,
                key,
                query,
                value,
                edge_features=edge_features,
                dropout_mask=dropout_mask,
                num_heads=self.heads,
                concat_heads=self.concat,
                norm_by_dim=True,
                return_attention_weights=return_attention,
                deterministic=deterministic,
            )
            if return_attention:
                out, attention = result
                attention = attention.t().to(x_src.dtype)
            else:
                out = result

        if self.root_weight:
            x_r = self.lin_skip(x_dst)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if return_attention:
            return out, (output_edge_index, attention)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
