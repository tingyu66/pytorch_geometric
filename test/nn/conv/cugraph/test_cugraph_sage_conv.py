import pytest
import torch

from torch_geometric import EdgeIndex
from torch_geometric.nn import CuGraphSAGEConv, SAGEConv
from torch_geometric.testing import onlyCUDA, withPackage

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
INDEX_DTYPES = [torch.int32, torch.int64]


def assert_close(actual: torch.Tensor, expected: torch.Tensor):
    tolerances = {
        torch.float32: (1e-6, 1e-5),
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (5e-2, 5e-2),
    }
    atol, rtol = tolerances[actual.dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
                               equal_nan=True)


def copy_sage_weights(
    source: SAGEConv,
    target: CuGraphSAGEConv,
    src_channels: int,
    dst_channels: int,
):
    with torch.no_grad():
        if source.project:
            target.pre_lin.weight.copy_(source.lin.weight)
            target.pre_lin.bias.copy_(source.lin.bias)
        target.lin.weight[:, :src_channels].copy_(source.lin_l.weight)
        if source.root_weight:
            target.lin.weight[:, src_channels:src_channels +
                              dst_channels].copy_(source.lin_r.weight)
        if source.lin_l.bias is not None:
            target.lin.bias.copy_(source.lin_l.bias)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max'])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('bipartite', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('project', [True, False])
@pytest.mark.parametrize('root_weight', [True, False])
@pytest.mark.parametrize('dtype', DTYPES)
def test_sage_conv_equality(aggr, bias, bipartite, normalize, project,
                            root_weight, dtype):
    src_channels = 8
    dst_channels = 4 if bipartite else 8
    in_channels = (src_channels, dst_channels) if bipartite else src_channels
    out_channels = 16
    kwargs = dict(aggr=aggr, bias=bias, normalize=normalize, project=project,
                  root_weight=root_weight)

    node_offsets = torch.arange(10, device='cuda').view(-1, 1) * 0.1
    x_src1 = (node_offsets +
              torch.rand(10, src_channels, device='cuda') * 0.025).to(dtype)
    x_src1.requires_grad_(True)
    x_src2 = x_src1.detach().clone().requires_grad_(True)
    if bipartite:
        x_dst1 = torch.rand(8, dst_channels, device='cuda', dtype=dtype,
                            requires_grad=True)
        x_dst2 = x_dst1.detach().clone().requires_grad_(True)
        x1 = (x_src1, x_dst1)
        x2 = (x_src2, x_dst2)
        size = (10, 8)
    else:
        x1 = x_src1
        x2 = x_src2
        size = (10, 10)

    edge_index = torch.tensor([
        [7, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ], device='cuda')

    conv1 = SAGEConv(in_channels, out_channels,
                     **kwargs).to(device='cuda', dtype=dtype)
    conv2 = CuGraphSAGEConv(in_channels, out_channels,
                            **kwargs).to(device='cuda', dtype=dtype)
    copy_sage_weights(conv1, conv2, src_channels, dst_channels)

    expected_in_channels = (src_channels +
                            dst_channels if root_weight else src_channels)
    assert conv2.lin.weight.size(1) == expected_in_channels

    out1 = conv1(x1, edge_index, size=size)
    out2 = conv2(x2, edge_index, size=size)
    assert_close(out1, out2)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    assert_close(x_src1.grad, x_src2.grad)
    if bipartite:
        if root_weight:
            assert_close(x_dst1.grad, x_dst2.grad)
        else:
            assert x_dst1.grad is None
            assert x_dst2.grad is None

    if project:
        assert_close(conv1.lin.weight.grad, conv2.pre_lin.weight.grad)
        assert_close(conv1.lin.bias.grad, conv2.pre_lin.bias.grad)

    assert_close(conv1.lin_l.weight.grad,
                 conv2.lin.weight.grad[:, :src_channels])
    if root_weight:
        assert_close(
            conv1.lin_r.weight.grad,
            conv2.lin.weight.grad[:, src_channels:src_channels + dst_channels],
        )
    if bias:
        assert_close(conv1.lin_l.bias.grad, conv2.lin.bias.grad)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('index_dtype', INDEX_DTYPES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_sage_conv_without_target_features(index_dtype, dtype):
    src_channels, dst_channels = 8, 16
    x1 = torch.rand(4, src_channels, device='cuda', dtype=dtype,
                    requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 1]],
        device='cuda',
        dtype=index_dtype,
    )

    conv1 = SAGEConv((src_channels, dst_channels),
                     32).to(device='cuda', dtype=dtype)
    conv2 = CuGraphSAGEConv((src_channels, dst_channels),
                            32).to(device='cuda', dtype=dtype)
    copy_sage_weights(conv1, conv2, src_channels, dst_channels)

    out1 = conv1((x1, None), edge_index, size=(4, 2))
    out2 = conv2((x2, None), edge_index, size=(4, 2))
    assert_close(out1, out2)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert_close(x1.grad, x2.grad)
    assert_close(conv1.lin_l.weight.grad,
                 conv2.lin.weight.grad[:, :src_channels])
    assert torch.count_nonzero(
        conv2.lin.weight.grad[:, src_channels:]).item() == 0
    assert_close(conv1.lin_l.bias.grad, conv2.lin.bias.grad)


@onlyCUDA
@withPackage('cudnn')
def test_lazy_sage_conv():
    x = torch.rand(4, 8, device='cuda')
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device='cuda')

    conv = CuGraphSAGEConv(-1, 32).cuda()
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    assert conv.lin.weight.size() == (32, 16)

    with pytest.raises(ValueError, match="does not support lazy"):
        CuGraphSAGEConv(-1, 32, project=True)

    conv = CuGraphSAGEConv((-1, -1), 32).cuda()
    with pytest.raises(ValueError, match="without target features"):
        conv((x, None), edge_index, size=(4, 2))


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('index_dtype', INDEX_DTYPES)
def test_sage_conv_edge_index(index_dtype):
    x = torch.rand(4, 8, device='cuda')
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 1]],
        device='cuda',
        dtype=index_dtype,
    )
    csc = EdgeIndex(edge_index, sparse_size=(4, 2), sort_order='col')
    conv = CuGraphSAGEConv((8, 8), 32).cuda()

    out1 = conv((x, x[:2]), edge_index, size=(4, 2))
    out2 = conv((x, x[:2]), csc)
    assert_close(out1, out2)


@onlyCUDA
@withPackage('cudnn')
def test_sage_conv_unsupported_dtype():
    x = torch.rand(4, 8, device='cuda', dtype=torch.float64)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device='cuda')
    conv = CuGraphSAGEConv(8, 32).to(device='cuda', dtype=torch.float64)

    with pytest.raises(TypeError, match='float32, float16, or bfloat16'):
        conv(x, edge_index)


@onlyCUDA
@withPackage('cudnn')
def test_sage_conv_invalid_aggr():
    with pytest.raises(ValueError, match="Aggregation function"):
        CuGraphSAGEConv(8, 32, aggr='lstm')
