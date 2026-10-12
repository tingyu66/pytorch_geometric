import pytest
import torch

from torch_geometric import EdgeIndex
from torch_geometric.nn import CuGraphTransformerConv, TransformerConv
from torch_geometric.testing import onlyCUDA, withPackage

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
INDEX_DTYPES = [torch.int32, torch.int64]


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    gradient: bool = False,
):
    if gradient:
        tolerances = {
            torch.float32: (7e-3, 7e-3),
            torch.float16: (7e-2, 7e-2),
            torch.bfloat16: (7e-2, 7e-2),
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
@pytest.mark.parametrize('index_dtype', INDEX_DTYPES)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize(
    'bias,concat,beta,edge_dim,root_weight',
    [
        (True, True, False, None, True),
        (False, False, False, None, True),
        (True, True, True, 3, True),
        (False, False, True, 3, False),
    ],
)
def test_transformer_conv_equality(index_dtype, dtype, bias, concat, beta,
                                   edge_dim, root_weight):
    torch.manual_seed(12345)
    in_channels, out_channels, heads = 8, 4, 2
    x1 = torch.randn(
        6, in_channels, device='cuda', dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    edge_index = torch.tensor([
        [4, 0, 2, 1, 5, 3, 0, 2],
        [1, 2, 0, 1, 2, 0, 0, 2],
    ], device='cuda', dtype=index_dtype)
    csc = EdgeIndex(edge_index, sparse_size=(6, 6)).sort_by('col')[0]
    (colptr, row), _ = csc.get_csc()
    assert colptr.dtype == row.dtype == index_dtype
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
        beta=beta,
        edge_dim=edge_dim,
        bias=bias,
        root_weight=root_weight,
    )
    conv1 = TransformerConv(in_channels, out_channels, **kwargs).to(
        device='cuda', dtype=dtype)
    conv2 = CuGraphTransformerConv(in_channels, out_channels, **kwargs).to(
        device='cuda', dtype=dtype)
    conv2.load_state_dict(conv1.state_dict())

    out1, (returned_edge_index1, attention1) = conv1(
        x1, edge_index, edge_attr1, return_attention_weights=True)
    out2, (returned_edge_index2, attention2) = conv2(
        x2, edge_index, edge_attr2, return_attention_weights=True)

    assert torch.equal(returned_edge_index1, returned_edge_index2)
    assert_close(out1, out2)
    assert_close(attention1, attention2)

    grad_out = torch.randn_like(out1)
    grad_attention = torch.randn_like(attention1)
    torch.autograd.backward((out1, attention1),
                            (grad_out, grad_attention))
    torch.autograd.backward((out2, attention2),
                            (grad_out, grad_attention))

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
@pytest.mark.parametrize('index_dtype', INDEX_DTYPES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_transformer_conv_bipartite_equality(index_dtype, dtype):
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
    ], device='cuda', dtype=index_dtype)
    csc = EdgeIndex(edge_index, sparse_size=(4, 6)).sort_by('col')[0]
    (colptr, row), _ = csc.get_csc()
    assert colptr.dtype == row.dtype == index_dtype
    edge_attr1 = torch.randn(
        edge_index.size(1), edge_dim, device='cuda', dtype=dtype,
        requires_grad=True)
    edge_attr2 = edge_attr1.detach().clone().requires_grad_()

    kwargs = dict(
        heads=heads,
        concat=False,
        beta=True,
        edge_dim=edge_dim,
    )
    conv1 = TransformerConv(in_channels, out_channels, **kwargs).to(
        device='cuda', dtype=dtype)
    conv2 = CuGraphTransformerConv(in_channels, out_channels, **kwargs).to(
        device='cuda', dtype=dtype)
    conv2.load_state_dict(conv1.state_dict())

    out1, (_, attention1) = conv1(
        (x_src1, x_dst1), edge_index, edge_attr1,
        return_attention_weights=True)
    out2, (_, attention2) = conv2(
        (x_src2, x_dst2), edge_index, edge_attr2,
        return_attention_weights=True)
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


@onlyCUDA
@withPackage('cudnn')
def test_transformer_conv_returns_attention_for_false():
    x = torch.randn(4, 8, device='cuda')
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ], device='cuda')
    conv = CuGraphTransformerConv(8, 4, heads=2).cuda()

    out, (returned_edge_index, attention) = conv(
        x, edge_index, return_attention_weights=False)
    assert out.size() == (4, 8)
    assert torch.equal(returned_edge_index, edge_index)
    assert attention.size() == (edge_index.size(1), 2)


@onlyCUDA
@withPackage('cudnn')
def test_transformer_conv_dropout():
    x = torch.randn(4, 8, device='cuda')
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ], device='cuda')
    conv1 = TransformerConv(
        8, 4, heads=2, dropout=0.5, root_weight=False).cuda()
    conv2 = CuGraphTransformerConv(
        8, 4, heads=2, dropout=0.5, root_weight=False).cuda()
    conv2.load_state_dict(conv1.state_dict())

    torch.manual_seed(24680)
    out1, (_, attention1) = conv1(
        x, edge_index, return_attention_weights=True)
    torch.manual_seed(24680)
    out2, (_, attention2) = conv2(
        x, edge_index, return_attention_weights=True)
    assert_close(out1, out2)
    assert_close(attention1, attention2)


@onlyCUDA
@withPackage('cudnn')
def test_transformer_conv_lazy():
    x = torch.randn(4, 8, device='cuda')
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ], device='cuda')
    conv = CuGraphTransformerConv(-1, 4, heads=2).cuda()

    out = conv(x, edge_index)
    assert out.size() == (4, 8)
    assert conv.lin_key.weight.size() == (8, 8)
    assert conv.lin_query.weight.size() == (8, 8)
    assert conv.lin_value.weight.size() == (8, 8)


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('index_dtype', INDEX_DTYPES)
def test_transformer_conv_deterministic_backward(index_dtype):
    x = torch.randn(4, 8, device='cuda', requires_grad=True)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ], device='cuda', dtype=index_dtype)
    conv = CuGraphTransformerConv(8, 4, heads=2).cuda()

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        conv(x, edge_index).sum().backward()
    finally:
        torch.use_deterministic_algorithms(previous)
    assert x.grad is not None


@onlyCUDA
@withPackage('cudnn')
def test_transformer_conv_empty_edges():
    x1 = torch.randn(4, 8, device='cuda', requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    edge_index = torch.empty((2, 0), device='cuda', dtype=torch.long)
    conv1 = TransformerConv(8, 4, heads=2, beta=True).cuda()
    conv2 = CuGraphTransformerConv(8, 4, heads=2, beta=True).cuda()
    conv2.load_state_dict(conv1.state_dict())

    out1, (_, attention1) = conv1(
        x1, edge_index, return_attention_weights=True)
    out2, (_, attention2) = conv2(
        x2, edge_index, return_attention_weights=True)
    assert_close(out1, out2)
    assert attention1.size() == attention2.size() == (0, 2)

    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    assert_close(x1.grad, x2.grad, gradient=True)


@withPackage('cudnn')
def test_transformer_conv_rejects_cpu_input():
    conv = CuGraphTransformerConv((8, 4), 4)
    x_src = torch.randn(4, 8)
    x_dst = torch.randn(2, 4)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ])

    with pytest.raises(RuntimeError, match='requires GPU'):
        conv((x_src, x_dst), edge_index)
