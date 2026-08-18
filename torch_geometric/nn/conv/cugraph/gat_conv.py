from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric import EdgeIndex
from torch_geometric.nn.conv.cugraph import CuGraphModule
from torch_geometric.nn.conv.cugraph.base import LEGACY_MODE
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops

try:
    if LEGACY_MODE:
        from pylibcugraphops.torch.autograd import mha_gat_n2n as GATConvAgg
    else:
        from pylibcugraphops.pytorch.operators import mha_gat_n2n as GATConvAgg
except ImportError:
    pass


class CuGraphGATConv(CuGraphModule):  # pragma: no cover
    r'''The graph attentional operator from the Graph Attention Networks paper
    <https://arxiv.org/abs/1710.10903>.

    This is an optimized version of
    GATConv backed by cugraph-ops. Its
    constructor and forward arguments mirror GATConv, with the additional
    max_num_neighbors forward hint used by cugraph-ops.

    The fused implementation requires GPU-resident dense edge indices. Plain
    edge-index tensors are accepted and converted to EdgeIndex;
    sparse tensor layouts are not supported.
    '''
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
        **kwargs,
    ):
        aggr = kwargs.pop('aggr', 'add')
        node_dim = kwargs.pop('node_dim', 0)
        if aggr != 'add' or node_dim != 0 or kwargs:
            raise ValueError(
                'CuGraphGATConv only supports aggr=\'add\', node_dim=0, and '
                'no additional MessagePassing options')

        super().__init__()

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
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

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
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
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

    def _to_edge_index(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size,
    ) -> EdgeIndex:
        if isinstance(edge_index, EdgeIndex):
            if size is not None and tuple(edge_index.sparse_size()) != size:
                return EdgeIndex(edge_index.as_tensor(), sparse_size=size)
            return edge_index

        if not isinstance(edge_index,
                          Tensor) or edge_index.layout != torch.strided:
            raise ValueError(
                f'CuGraphGATConv requires a dense edge-index tensor '
                f'(got {type(edge_index)})')

        if size is None:
            if isinstance(x, Tensor):
                size = (x.size(0), x.size(0))
            else:
                src, dst = x
                num_dst = (dst.size(0)
                           if dst is not None else 0 if edge_index.numel() == 0
                           else int(edge_index[1].max()) + 1)
                size = (src.size(0), num_dst)

        return EdgeIndex(edge_index, sparse_size=size)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None,
        max_num_neighbors: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        r'''Runs the forward pass of the module.

        In addition to the native GATConv arguments,
        max_num_neighbors may provide the maximum destination in-degree for
        bipartite graphs. Attention weights are returned in the same edge order
        as edge_index.
        '''
        edge_index = self._to_edge_index(x, edge_index, size)
        num_src_nodes, num_dst_nodes = edge_index.sparse_size()

        if isinstance(x, Tensor):
            x_src_input = x_dst_input = x
        else:
            x_src_input, x_dst_input = x

        if self.add_self_loops:
            num_nodes = x_src_input.size(0)
            if x_dst_input is not None:
                num_nodes = min(num_nodes, x_dst_input.size(0))
            num_nodes = min(size) if size is not None else num_nodes
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                fill_value=self.fill_value,
                num_nodes=num_nodes,
            )
            if not isinstance(edge_index, EdgeIndex):
                edge_index = EdgeIndex(
                    edge_index, sparse_size=(num_src_nodes, num_dst_nodes))

        res: Optional[Tensor] = None
        if isinstance(x, Tensor):
            if self.res is not None:
                res = self.res(x)[:num_dst_nodes]

            if self.lin is not None:
                x_src = self.lin(x)
                x_dst = x_src[:num_dst_nodes]
            else:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x)
                x_dst = self.lin_dst(x)[:num_dst_nodes]
        else:
            x_src_input, x_dst_input = x
            if x_dst_input is not None and self.res is not None:
                res = self.res(x_dst_input)

            if self.lin is not None:
                x_src = self.lin(x_src_input)
                x_dst = (None
                         if x_dst_input is None else self.lin(x_dst_input))
            else:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x_src_input)
                x_dst = (None
                         if x_dst_input is None else self.lin_dst(x_dst_input))

        use_bipartite = (num_src_nodes != num_dst_nodes
                         or not isinstance(x, Tensor) or self.lin is None)
        if x_dst is None:
            x_dst = x_src.new_zeros(
                (num_dst_nodes, self.heads * self.out_channels))

        graph = self.get_cugraph(
            edge_index,
            max_num_neighbors,
            is_bipartite=use_bipartite,
            preserve_edge_order=True,
        )

        edge_feat = None
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_feat = self.lin_edge(edge_attr)

        attention = [self.att_src.reshape(-1), self.att_dst.reshape(-1)]
        if edge_feat is not None:
            assert self.att_edge is not None
            attention.append(self.att_edge.reshape(-1))
        attn_weights = torch.cat(attention)

        return_attention = bool(return_attention_weights)
        if LEGACY_MODE:
            if ((self.training and self.dropout > 0.0) or return_attention):
                raise NotImplementedError(
                    'dropout and return_attention_weights require the modern '
                    'pylibcugraphops API')
            feat = (x_src, x_dst) if use_bipartite else x_src
            out = GATConvAgg(feat, attn_weights, graph, self.heads,
                             'LeakyReLU', self.negative_slope, False,
                             self.concat, edge_feat=edge_feat)
            alpha = None
        else:
            feat = (x_src, x_dst) if use_bipartite else x_src
            result = GATConvAgg(
                feat,
                attn_weights,
                graph,
                self.heads,
                'LeakyReLU',
                self.negative_slope,
                self.concat,
                edge_feat=edge_feat,
                dropout=self.dropout,
                training=self.training,
                return_attention_weights=return_attention,
            )
            if return_attention:
                out, alpha = result
            else:
                out = result
                alpha = None

        if res is not None:
            out = out + res
        if self.bias is not None:
            out = out + self.bias

        if return_attention:
            assert alpha is not None
            return out, (edge_index, alpha)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
