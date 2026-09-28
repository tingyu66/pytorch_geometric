from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric import EdgeIndex
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptPairTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops

try:
    from cudnn.gnn import CscGraph, mha_gat
    HAS_CUDNN_GNN = True
except ImportError:
    HAS_CUDNN_GNN = False


class CuGraphGATConv(torch.nn.Module):  # pragma: no cover
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    :class:`CuGraphGATConv` is a CUDA-optimized version of
    :class:`~torch_geometric.nn.conv.GATConv` that uses the cuDNN GNN API for
    fused multi-head attention on homogeneous or bipartite graphs.

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
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of normalized attention
            coefficients. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality.
            (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): How to generate
            edge features for added self-loops. (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        residual (bool, optional): If set to :obj:`True`, the layer adds a
            learnable skip connection. (default: :obj:`False`)

    Install :obj:`nvidia-cudnn-frontend` with a cuDNN 9.28 or newer backend to
    use this operator.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        residual: bool = False,
    ):
        super().__init__()

        if not HAS_CUDNN_GNN:
            raise ModuleNotFoundError(
                f"'{self.__class__.__name__}' requires "
                f"'nvidia-cudnn-frontend' and cuDNN >= 9.28")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels,
                                  bias=False, weight_initializer='glorot')
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        total_out_channels = out_channels * (heads if concat else 1)
        if residual:
            res_in_channels = (in_channels if isinstance(in_channels, int)
                               else in_channels[1])
            self.res = Linear(res_in_channels, total_out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        x_src_input, x_dst_input = (x, x) if isinstance(x, Tensor) else x
        if x_src_input.dim() != 2 or (x_dst_input is not None
                                     and x_dst_input.dim() != 2):
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"static graphs")
        if not x_src_input.is_cuda or (x_dst_input is not None
                                       and not x_dst_input.is_cuda):
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        if size is None:
            sparse_size = (None, None)
            if isinstance(edge_index, EdgeIndex):
                sparse_size = edge_index.sparse_size()
            num_src_nodes = sparse_size[0] or x_src_input.size(0)
            num_dst_nodes = sparse_size[1]
            if num_dst_nodes is None:
                if x_dst_input is not None:
                    num_dst_nodes = x_dst_input.size(0)
                elif edge_index.numel() > 0:
                    num_dst_nodes = int(edge_index[1].max()) + 1
                else:
                    num_dst_nodes = 0
            size = (num_src_nodes, num_dst_nodes)

        if size[0] != x_src_input.size(0):
            raise ValueError(f"Expected {size[0]} source node features (got "
                             f"{x_src_input.size(0)})")
        if size[1] is None:
            raise ValueError("The number of destination nodes must be known")
        if isinstance(x, Tensor):
            x_dst_input = x[:size[1]]
        if x_dst_input is not None and size[1] != x_dst_input.size(0):
            raise ValueError(f"Expected {size[1]} destination node features "
                             f"(got {x_dst_input.size(0)})")

        if self.add_self_loops:
            num_nodes = min(size)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                fill_value=self.fill_value,
                num_nodes=num_nodes,
            )

        output_edge_index = edge_index
        edge_index = EdgeIndex(edge_index, sparse_size=size)
        edge_index, perm = edge_index.sort_by('col', stable=True)
        (colptr, row), _ = edge_index.get_csc()

        graph = CscGraph(colptr, row, num_src_nodes=size[0],
                         map_csc_to_coo=perm)
        deterministic = torch.are_deterministic_algorithms_enabled()
        if deterministic:
            graph = graph.with_reverse_csc()

        if self.lin is not None:
            src_features = self.lin(x_src_input)
            dst_features = (None if x_dst_input is None else
                            self.lin(x_dst_input))
        else:
            assert self.lin_src is not None and self.lin_dst is not None
            src_features = self.lin_src(x_src_input)
            dst_features = (None if x_dst_input is None else
                            self.lin_dst(x_dst_input))
        if dst_features is None:
            dst_features = src_features.new_zeros(
                (size[1], self.heads * self.out_channels))

        attn_weights = [self.att_src.flatten(), self.att_dst.flatten()]
        edge_features = None
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_features = self.lin_edge(edge_attr)
            assert self.att_edge is not None
            attn_weights.append(self.att_edge.flatten())
        weights = torch.cat(attn_weights)

        dropout_mask = None
        if self.training and self.dropout > 0.0:
            dropout_mask = F.dropout(
                torch.ones(
                    (self.heads, row.numel()),
                    device=x_src_input.device,
                    dtype=torch.float32,
                ),
                p=self.dropout,
                training=True,
            )

        return_attention = bool(return_attention_weights)
        if row.numel() == 0:
            out = dst_features * 0
            out = out + src_features.sum() * 0 + weights.sum() * 0
            if not self.concat:
                out = out.view(-1, self.heads, self.out_channels).mean(dim=1)
            attention = torch.empty(
                (0, self.heads),
                device=x_src_input.device,
                dtype=x_src_input.dtype,
            )
        else:
            result = mha_gat(
                graph,
                (src_features, dst_features),
                weights,
                edge_features=edge_features,
                dropout_mask=dropout_mask,
                num_heads=self.heads,
                concat_heads=self.concat,
                activation='leaky_relu',
                activation_alpha=self.negative_slope,
                return_attention_weights=return_attention,
                deterministic=deterministic,
            )
            if return_attention:
                out, attention = result
                attention = attention.t().to(x_src_input.dtype)
            else:
                out = result

        if self.res is not None and x_dst_input is not None:
            out = out + self.res(x_dst_input)
        if self.bias is not None:
            out = out + self.bias

        if return_attention:
            return out, (output_edge_index, attention)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
