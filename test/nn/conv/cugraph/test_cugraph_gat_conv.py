import pytest
import torch

from torch_geometric import EdgeIndex
from torch_geometric.nn import CuGraphGATConv, GATConv
from torch_geometric.testing import onlyCUDA, withPackage

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    gradient: bool = False,
):
    if gradient:
        tolerances = {
            torch.float32: (6e-3, 6e-3),
            torch.float16: (6e-2, 6e-2),
            torch.bfloat16: (6e-2, 6e-2),
        }
    else:
        tolerances = {
            torch.float32: (5e-4, 5e-4),
            torch.float16: (2e-2, 2e-2),
            torch.bfloat16: (2e-2, 2e-2),
        }
    atol, rtol = tolerances[actual.dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize(
    'bias,concat,edge_dim,residual,add_self_loops',
    [
        (True, True, None, False, False),
        (False, False, None, True, True),
        (True, True, 3, True, True),
        (False, False, 3, False, False),
    ],
)
def test_gat_conv_equality(dtype, bias, concat, edge_dim, residual,
                           add_self_loops):
    torch.manual_seed(12345)
    in_channels, out_channels, heads = 8, 4, 2
    x1 = torch.randn(
        6,
        in_channels,
        device='cuda',
        dtype=dtype,
        requires_grad=True,
    )
    x2 = x1.detach().clone().requires_grad_()
    edge_index = torch.tensor([
        [4, 0, 2, 1, 5, 3, 0, 2],
        [1, 2, 0, 1, 2, 0, 0, 2],
    ], device='cuda')
    edge_attr1 = None
    edge_attr2 = None
    if edge_dim is not None:
        edge_attr1 = torch.randn(
            edge_index.size(1),
            edge_dim,
            device='cuda',
            dtype=dtype,
            requires_grad=True,
        )
        edge_attr2 = edge_attr1.detach().clone().requires_grad_()

    kwargs = dict(
        heads=heads,
        concat=concat,
        add_self_loops=add_self_loops,
        edge_dim=edge_dim,
        bias=bias,
        residual=residual,
    )
    conv1 = GATConv(
        in_channels,
        out_channels,
        **kwargs,
    ).to(device='cuda', dtype=dtype)
    conv2 = CuGraphGATConv(
        in_channels,
        out_channels,
        **kwargs,
    ).to(device='cuda', dtype=dtype)
    conv2.load_state_dict(conv1.state_dict())

    out1, (returned_edge_index1, attention1) = conv1(
        x1,
        edge_index,
        edge_attr1,
        return_attention_weights=True,
    )
    out2, (returned_edge_index2, attention2) = conv2(
        x2,
        edge_index,
        edge_attr2,
        return_attention_weights=True,
    )

    assert torch.equal(returned_edge_index1, returned_edge_index2)
    assert_close(out1, out2)
    assert_close(attention1, attention2)

    grad_out = torch.randn_like(out1)
    grad_attention = torch.randn_like(attention1)
    torch.autograd.backward((out1, attention1), (grad_out, grad_attention))
    torch.autograd.backward((out2, attention2), (grad_out, grad_attention))

    assert_close(x1.grad, x2.grad, gradient=True)
    if edge_attr1 is not None:
        assert edge_attr2 is not None
        assert_close(edge_attr1.grad, edge_attr2.grad, gradient=True)

    params1 = dict(conv1.named_parameters())
    params2 = dict(conv2.named_parameters())
    assert params1.keys() == params2.keys()
    for name in params1:
        grad1 = params1[name].grad
        grad2 = params2[name].grad
        if grad1 is None:
            assert grad2 is None
        else:
            assert grad2 is not None
            assert_close(grad1, grad2, gradient=True)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('dtype', DTYPES)
def test_gat_conv_bipartite_equality(dtype):
    torch.manual_seed(13579)
    in_channels = (8, 5)
    out_channels, heads, edge_dim = 4, 2, 3
    x_src1 = torch.randn(
        4, in_channels[0], device='cuda', dtype=dtype, requires_grad=True)
    x_dst1 = torch.randn(
        6, in_channels[1], device='cuda', dtype=dtype, requires_grad=True)
    x_src2 = x_src1.detach().clone().requires_grad_()
    x_dst2 = x_dst1.detach().clone().requires_grad_()
    edge_index = torch.tensor([
        [0, 1, 3, 2, 0, 3, 1, 2],
        [0, 1, 2, 3, 4, 5, 5, 4],
    ], device='cuda')
    edge_attr1 = torch.randn(
        edge_index.size(1), edge_dim, device='cuda', dtype=dtype,
        requires_grad=True)
    edge_attr2 = edge_attr1.detach().clone().requires_grad_()

    kwargs = dict(
        heads=heads,
        concat=False,
        add_self_loops=False,
        edge_dim=edge_dim,
        residual=True,
    )
    conv1 = GATConv(in_channels, out_channels, **kwargs).to(
        device='cuda', dtype=dtype)
    conv2 = CuGraphGATConv(in_channels, out_channels, **kwargs).to(
        device='cuda', dtype=dtype)
    conv2.load_state_dict(conv1.state_dict())

    out1, (returned_edge_index1, attention1) = conv1(
        (x_src1, x_dst1), edge_index, edge_attr1,
        return_attention_weights=True)
    out2, (returned_edge_index2, attention2) = conv2(
        (x_src2, x_dst2), edge_index, edge_attr2,
        return_attention_weights=True)

    assert out1.size(0) == x_dst1.size(0) > x_src1.size(0)
    assert torch.equal(returned_edge_index1, returned_edge_index2)
    assert_close(out1, out2)
    assert_close(attention1, attention2)

    grad_out = torch.randn_like(out1)
    grad_attention = torch.randn_like(attention1)
    torch.autograd.backward((out1, attention1),
                            (grad_out, grad_attention))
    torch.autograd.backward((out2, attention2),
                            (grad_out, grad_attention))

    assert_close(x_src1.grad, x_src2.grad, gradient=True)
    assert_close(x_dst1.grad, x_dst2.grad, gradient=True)
    assert_close(edge_attr1.grad, edge_attr2.grad, gradient=True)

    params1 = dict(conv1.named_parameters())
    params2 = dict(conv2.named_parameters())
    assert params1.keys() == params2.keys()
    for name in params1:
        assert params1[name].grad is not None
        assert params2[name].grad is not None
        assert_close(params1[name].grad, params2[name].grad, gradient=True)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('dtype', DTYPES)
def test_gat_conv_homogeneous_prefix(dtype):
    torch.manual_seed(23456)
    x1 = torch.randn(
        6,
        8,
        device='cuda',
        dtype=dtype,
        requires_grad=True,
    )
    x2 = x1.detach().clone().requires_grad_()
    edge_index = torch.tensor([
        [5, 0, 1, 2, 3, 4],
        [0, 1, 1, 2, 3, 3],
    ], device='cuda')
    size = (6, 4)

    conv1 = GATConv(
        8,
        4,
        heads=2,
        add_self_loops=False,
    ).to(device='cuda', dtype=dtype)
    conv2 = CuGraphGATConv(
        8,
        4,
        heads=2,
        add_self_loops=False,
    ).to(device='cuda', dtype=dtype)
    conv2.load_state_dict(conv1.state_dict())

    out1 = conv1((x1, x1[:size[1]]), edge_index, size=size)
    out2 = conv2(x2, edge_index, size=size)
    assert_close(out1, out2)

    grad_out = torch.randn_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert_close(x1.grad, x2.grad, gradient=True)


@onlyCUDA
@withPackage('cudnn')
def test_gat_conv_edge_index():
    torch.manual_seed(34567)
    x = torch.randn(6, 8, device='cuda')
    edge_index = torch.tensor([
        [4, 0, 2, 1, 5, 3, 0, 2],
        [1, 2, 0, 1, 2, 0, 0, 2],
    ], device='cuda')
    csc = EdgeIndex(
        edge_index,
        sparse_size=(6, 6),
    ).sort_by('col', stable=True)[0]

    conv = CuGraphGATConv(8, 4, heads=2,
                          add_self_loops=False).cuda()
    out1, (_, attention1) = conv(
        x,
        edge_index,
        return_attention_weights=True,
    )
    out2, (_, attention2) = conv(
        x,
        csc,
        return_attention_weights=True,
    )
    assert_close(out1, out2)

    perm = EdgeIndex(
        edge_index,
        sparse_size=(6, 6),
    ).sort_by('col', stable=True)[1]
    assert perm is not None
    assert_close(attention1[perm], attention2)


@onlyCUDA
@withPackage('cudnn')
def test_gat_conv_dropout():
    x = torch.randn(4, 8, device='cuda')
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ], device='cuda')
    conv = CuGraphGATConv(
        8,
        4,
        heads=2,
        dropout=1.0,
        add_self_loops=False,
        bias=False,
    ).cuda()
    conv.train()

    out, (_, attention) = conv(
        x,
        edge_index,
        return_attention_weights=True,
    )
    assert torch.count_nonzero(out).item() == 0
    assert torch.count_nonzero(attention).item() == 0


@onlyCUDA
@withPackage('cudnn')
def test_gat_conv_lazy():
    x = torch.randn(4, 8, device='cuda')
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ], device='cuda')
    conv = CuGraphGATConv(-1, 4, heads=2,
                          add_self_loops=False).cuda()

    out = conv(x, edge_index)
    assert out.size() == (4, 8)
    assert conv.lin.weight.size() == (8, 8)


@withPackage('cudnn')
def test_gat_conv_rejects_cpu_input():
    conv = CuGraphGATConv((8, 4), 4, add_self_loops=False)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ])
    with pytest.raises(RuntimeError, match='requires GPU'):
        conv(x, edge_index)


@onlyCUDA
@withPackage('cudnn')
def test_gat_conv_unsupported_dtype():
    x = torch.randn(4, 8, device='cuda', dtype=torch.float64)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ], device='cuda')
    conv = CuGraphGATConv(
        8,
        4,
        add_self_loops=False,
    ).to(device='cuda', dtype=torch.float64)

    with pytest.raises(TypeError, match='float32, float16, or bfloat16'):
        conv(x, edge_index)
