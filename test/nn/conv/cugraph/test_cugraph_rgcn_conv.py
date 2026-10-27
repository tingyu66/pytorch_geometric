import pytest
import torch

from torch_geometric import EdgeIndex
from torch_geometric.nn import CuGraphRGCNConv
from torch_geometric.nn import FastRGCNConv as RGCNConv
from torch_geometric.testing import onlyCUDA, withPackage

INDEX_DTYPES = [torch.int32, torch.int64]


def assert_close(actual: torch.Tensor, expected: torch.Tensor):
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('index_dtype', INDEX_DTYPES)
@pytest.mark.parametrize('aggr', ['add', 'sum', 'mean'])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('bipartite', [True, False])
@pytest.mark.parametrize('num_bases', [1, 2, None])
@pytest.mark.parametrize('root_weight', [True, False])
def test_rgcn_conv_equality(index_dtype, aggr, bias, bipartite, num_bases,
                            root_weight):
    in_channels, out_channels, num_relations = 4, 2, 3
    kwargs = dict(
        aggr=aggr,
        bias=bias,
        num_bases=num_bases,
        root_weight=root_weight,
    )
    size = (10, 8) if bipartite else (10, 10)
    x1 = torch.rand(
        size[0], in_channels, device='cuda', requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    edge_index = torch.tensor([
        [7, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ], device='cuda', dtype=index_dtype)
    edge_type = torch.tensor(
        [1, 2, 1, 0, 2, 1, 2, 0, 2, 2, 1, 1, 1, 2, 2],
        device='cuda',
    )

    torch.manual_seed(12345)
    conv1 = RGCNConv(
        in_channels, out_channels, num_relations, **kwargs).cuda()
    torch.manual_seed(12345)
    conv2 = CuGraphRGCNConv(
        in_channels, out_channels, num_relations, **kwargs).cuda()

    if bipartite:
        out1 = conv1((x1, x1[:size[1]]), edge_index, edge_type)
    else:
        out1 = conv1(x1, edge_index, edge_type)
    out2 = conv2(x2, edge_index, edge_type, size=size)
    assert_close(out1, out2)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert_close(x1.grad, x2.grad)

    end = -1 if root_weight else None
    assert_close(conv1.weight.grad, conv2.weight.grad[:end])
    if root_weight:
        assert conv1.root is not None and conv1.root.grad is not None
        assert_close(conv1.root.grad, conv2.weight.grad[-1])
    if num_bases is not None:
        assert conv1.comp is not None and conv2.comp is not None
        assert_close(conv1.comp.grad, conv2.comp.grad)
    if bias:
        assert conv1.bias is not None and conv2.bias is not None
        assert_close(conv1.bias.grad, conv2.bias.grad)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('index_dtype', INDEX_DTYPES)
def test_rgcn_conv_csc_edge_order(index_dtype):
    torch.manual_seed(23456)
    x = torch.rand(6, 4, device='cuda')
    edge_index = torch.tensor([
        [4, 0, 2, 1, 5, 3, 0, 2],
        [1, 2, 0, 1, 2, 0, 0, 2],
    ], device='cuda', dtype=index_dtype)
    edge_type = torch.tensor(
        [2, 0, 1, 2, 1, 0, 2, 0], device='cuda')
    sorted_edge_index, perm = EdgeIndex(
        edge_index,
        sparse_size=(6, 6),
    ).sort_by('col', stable=True)
    assert perm is not None

    conv = CuGraphRGCNConv(4, 3, 3, num_bases=2).cuda()
    out1 = conv(x, edge_index, edge_type)
    out2 = conv(x, sorted_edge_index, edge_type[perm])
    assert_close(out1, out2)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('num_bases', [2, None])
@pytest.mark.parametrize('root_weight', [True, False])
def test_rgcn_conv_empty_edges(num_bases, root_weight):
    x1 = torch.rand(4, 3, device='cuda', requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    edge_index = torch.empty((2, 0), device='cuda', dtype=torch.long)
    edge_type = torch.empty(0, device='cuda', dtype=torch.long)
    kwargs = dict(num_bases=num_bases, root_weight=root_weight)

    torch.manual_seed(34567)
    conv1 = RGCNConv(3, 2, 3, **kwargs).cuda()
    torch.manual_seed(34567)
    conv2 = CuGraphRGCNConv(3, 2, 3, **kwargs).cuda()

    out1 = conv1(x1, edge_index, edge_type)
    out2 = conv2(x2, edge_index, edge_type)
    assert_close(out1, out2)
    grad = torch.rand_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    assert_close(x1.grad, x2.grad)


@withPackage('cudnn')
def test_rgcn_conv_rejects_cpu_input():
    conv = CuGraphRGCNConv(4, 2, 3)
    x = torch.rand(4, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_type = torch.tensor([0, 1])
    with pytest.raises(RuntimeError, match='requires GPU'):
        conv(x, edge_index, edge_type)


@onlyCUDA
@withPackage('cudnn')
def test_rgcn_conv_rejects_unsupported_dtype():
    conv = CuGraphRGCNConv(4, 2, 3).half().cuda()
    x = torch.rand(4, 4, device='cuda', dtype=torch.float16)
    edge_index = torch.tensor([[0, 1], [1, 0]], device='cuda')
    edge_type = torch.tensor([0, 1], device='cuda')
    with pytest.raises(TypeError, match='requires FP32'):
        conv(x, edge_index, edge_type)


@onlyCUDA
@withPackage('cudnn')
def test_rgcn_conv_rejects_deterministic_mode():
    conv = CuGraphRGCNConv(4, 2, 3).cuda()
    x = torch.rand(4, 4, device='cuda')
    edge_index = torch.tensor([[0, 1], [1, 0]], device='cuda')
    edge_type = torch.tensor([0, 1], device='cuda')
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(RuntimeError, match='deterministic backward'):
            conv(x, edge_index, edge_type)
    finally:
        torch.use_deterministic_algorithms(previous)


@withPackage('cudnn')
def test_rgcn_conv_invalid_options():
    with pytest.raises(ValueError, match='Aggregation function'):
        CuGraphRGCNConv(4, 2, 3, aggr='max')
    with pytest.raises(ValueError, match="'num_bases' must be positive"):
        CuGraphRGCNConv(4, 2, 3, num_bases=0)
