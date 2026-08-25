from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from torch_geometric import EdgeIndex
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor, Size

try:
    from cudnn.gnn import CscGraph, agg_simple
    HAS_CUDNN_GNN = True
except ImportError:
    HAS_CUDNN_GNN = False


class CuGraphSAGEConv(torch.nn.Module):  # pragma: no cover
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    :class:`CuGraphSAGEConv` is a CUDA-optimized version of
    :class:`~torch_geometric.nn.conv.SAGEConv` that uses the cuDNN GNN API for
    neighborhood aggregation and concatenates root features before applying a
    single linear transformation.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target features.
        out_channels (int): Size of each output sample.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"mean"`, :obj:`"sum"`, :obj:`"min"`, or :obj:`"max"`).
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features are
            :math:`\ell_2`-normalized. (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, transformed root
            node features are not added to the output. (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, source features are
            transformed and activated before aggregation. (default:
            :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer does not learn
            an additive bias. (default: :obj:`True`)

    Install :obj:`nvidia-cudnn-frontend` with a cuDNN 9.26 or newer backend to
    use this operator.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        if not HAS_CUDNN_GNN:
            raise ModuleNotFoundError(
                f"'{self.__class__.__name__}' requires "
                f"'nvidia-cudnn-frontend' and cuDNN >= 9.26")

        if aggr not in ['mean', 'sum', 'min', 'max']:
            raise ValueError(f"Aggregation function must be either 'mean', "
                             f"'sum', 'min' or 'max' (got '{aggr}')")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.pre_lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.root_weight:
            fused_in_channels = (in_channels[0] +
                                 in_channels[1] if in_channels[0] > 0
                                 and in_channels[1] > 0 else -1)
        else:
            fused_in_channels = in_channels[0]
        self.lin = Linear(fused_in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.pre_lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        x_src, x_dst = x

        if size is None:
            sparse_size = (None, None)
            if isinstance(edge_index, EdgeIndex):
                sparse_size = edge_index.sparse_size()

            num_src_nodes = sparse_size[0]
            if num_src_nodes is None:
                num_src_nodes = x_src.size(0)

            num_dst_nodes = sparse_size[1]
            if num_dst_nodes is None and x_dst is not None:
                num_dst_nodes = x_dst.size(0)

            size = (num_src_nodes, num_dst_nodes)

        edge_index = EdgeIndex(edge_index, sparse_size=size)
        edge_index = edge_index.sort_by('col')[0]
        num_src_nodes = edge_index.get_sparse_size(0)
        (colptr, row), _ = edge_index.get_csc()

        if not row.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        graph = CscGraph(colptr, row, num_src_nodes)

        if self.project:
            x_src = self.pre_lin(x_src).relu()

        if self.root_weight and x_dst is not None:
            out = agg_simple(
                graph,
                node_features=x_src,
                concat_features=x_dst,
                aggr=self.aggr,
            )
            out = self.lin(out)
        else:
            out = agg_simple(graph, node_features=x_src, aggr=self.aggr)
            if self.root_weight:
                if isinstance(self.lin.weight, UninitializedParameter):
                    raise ValueError(
                        f"'{self.__class__.__name__}' cannot initialize lazy "
                        f"root weights without target features")
                out = F.linear(
                    out,
                    self.lin.weight[:, :x_src.size(-1)],
                    self.lin.bias,
                )
            else:
                out = self.lin(out)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
