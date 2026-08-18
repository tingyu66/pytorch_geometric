import inspect

import pytest
import torch

from torch_geometric import EdgeIndex
from torch_geometric.nn import CuGraphGATConv, GATConv
from torch_geometric.testing import onlyCUDA, withPackage


def clone_input(x):
    if isinstance(x, tuple):
        values = []
        for value in x:
            cloned = (None if value is None else
                      value.detach().clone().requires_grad_())
            values.append(cloned)
        return tuple(values)
    return x.detach().clone().requires_grad_()


def assert_parameter_grads_equal(conv1, conv2):
    params1 = dict(conv1.named_parameters())
    params2 = dict(conv2.named_parameters())
    assert params1.keys() == params2.keys()
    for name in params1:
        torch.testing.assert_close(
            params1[name].grad,
            params2[name].grad,
            atol=2e-5,
            rtol=1e-5,
        )


def assert_input_grads_equal(x1, x2):
    if isinstance(x1, tuple):
        for value1, value2 in zip(x1, x2):
            if value1 is not None:
                torch.testing.assert_close(value1.grad, value2.grad, atol=2e-5,
                                           rtol=1e-5)
    else:
        torch.testing.assert_close(x1.grad, x2.grad, atol=2e-5, rtol=1e-5)


def as_tensor(edge_index):
    if isinstance(edge_index, EdgeIndex):
        return edge_index.as_tensor()
    return edge_index


def test_gat_conv_interface():
    native_init = inspect.signature(GATConv.__init__).parameters
    cugraph_init = inspect.signature(CuGraphGATConv.__init__).parameters
    assert tuple(native_init) == tuple(cugraph_init)

    native_forward = inspect.signature(GATConv.forward).parameters
    cugraph_forward = inspect.signature(CuGraphGATConv.forward).parameters
    assert tuple(native_forward) == tuple(cugraph_forward)[:-1]
    assert tuple(cugraph_forward)[-1] == 'max_num_neighbors'


@onlyCUDA
@withPackage('pylibcugraphops>=23.02')
@pytest.mark.parametrize('bipartite', [False, True])
@pytest.mark.parametrize('concat,bias', [(True, True), (False, False)])
def test_gat_conv_equality(bipartite, concat, bias):
    torch.manual_seed(12345)
    edge_index = torch.tensor([
        [4, 0, 5, 1, 3, 0, 2, 4],
        [1, 3, 0, 2, 1, 0, 3, 2],
    ], device='cuda')
    edge_attr = torch.randn(edge_index.size(1), 3, device='cuda')

    if bipartite:
        in_channels = (5, 7)
        size = (6, 4)
        x = (
            torch.randn(size[0], in_channels[0], device='cuda'),
            torch.randn(size[1], in_channels[1], device='cuda'),
        )
    else:
        in_channels = 5
        size = (6, 6)
        x = torch.randn(size[0], in_channels, device='cuda')

    kwargs = dict(
        heads=2,
        concat=concat,
        edge_dim=3,
        add_self_loops=False,
        bias=bias,
    )
    conv1 = GATConv(in_channels, 3, **kwargs).cuda()
    conv2 = CuGraphGATConv(in_channels, 3, **kwargs).cuda()
    assert conv1.state_dict().keys() == conv2.state_dict().keys()
    conv2.load_state_dict(conv1.state_dict())

    x1, x2 = clone_input(x), clone_input(x)
    edge_attr1 = edge_attr.detach().clone().requires_grad_()
    edge_attr2 = edge_attr.detach().clone().requires_grad_()
    out1 = conv1(x1, edge_index, edge_attr1, size=size)
    out2 = conv2(
        x2,
        edge_index,
        edge_attr2,
        size=size,
        max_num_neighbors=4,
    )
    torch.testing.assert_close(out1, out2, atol=2e-5, rtol=1e-5)

    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    assert_parameter_grads_equal(conv1, conv2)
    assert_input_grads_equal(x1, x2)
    torch.testing.assert_close(edge_attr1.grad, edge_attr2.grad, atol=2e-5,
                               rtol=1e-5)


@onlyCUDA
@withPackage('pylibcugraphops>=23.02')
def test_gat_conv_dropout_and_return_attention_weights():
    torch.manual_seed(23456)
    edge_index = EdgeIndex(
        torch.tensor([
            [2, 0, 4, 1, 3, 0, 5, 2, 1],
            [1, 3, 0, 4, 2, 1, 3, 4, 0],
        ], device='cuda'), sparse_size=(6, 6))
    x = torch.randn(6, 5, device='cuda')

    kwargs = dict(
        heads=2,
        dropout=0.35,
        add_self_loops=False,
        residual=True,
    )
    conv1 = GATConv(5, 3, **kwargs).cuda()
    conv2 = CuGraphGATConv(5, 3, **kwargs).cuda()
    conv2.load_state_dict(conv1.state_dict())
    conv1.train()
    conv2.train()

    x1, x2 = clone_input(x), clone_input(x)
    torch.manual_seed(34567)
    out1, (edges1, alpha1) = conv1(x1, edge_index,
                                   return_attention_weights=True)
    torch.manual_seed(34567)
    out2, (edges2, alpha2) = conv2(x2, edge_index,
                                   return_attention_weights=True)

    assert torch.equal(as_tensor(edges1), as_tensor(edges2))
    torch.testing.assert_close(alpha1, alpha2, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(out1, out2, atol=2e-5, rtol=1e-5)

    (out1.square().sum() + alpha1.square().sum()).backward()
    (out2.square().sum() + alpha2.square().sum()).backward()
    assert_parameter_grads_equal(conv1, conv2)
    assert_input_grads_equal(x1, x2)

    conv1.eval()
    conv2.eval()
    assert isinstance(conv1(x, edge_index, return_attention_weights=False),
                      torch.Tensor)
    assert isinstance(conv2(x, edge_index, return_attention_weights=False),
                      torch.Tensor)
    torch.testing.assert_close(conv1(x, edge_index), conv2(x, edge_index),
                               atol=2e-5, rtol=1e-5)


@onlyCUDA
@withPackage('pylibcugraphops>=23.02')
def test_gat_conv_self_loops_edge_features_and_residual():
    torch.manual_seed(45678)
    edge_index = torch.tensor([
        [2, 0, 4, 1, 3, 0, 2],
        [1, 3, 0, 4, 2, 1, 4],
    ], device='cuda')
    edge_attr = torch.randn(edge_index.size(1), 2, device='cuda')
    x = torch.randn(5, 4, device='cuda')

    kwargs = dict(
        heads=3,
        concat=False,
        dropout=0.2,
        edge_dim=2,
        fill_value='mean',
        residual=True,
    )
    conv1 = GATConv(4, 3, **kwargs).cuda().eval()
    conv2 = CuGraphGATConv(4, 3, **kwargs).cuda().eval()
    conv2.load_state_dict(conv1.state_dict())

    x1, x2 = clone_input(x), clone_input(x)
    edge_attr1 = edge_attr.detach().clone().requires_grad_()
    edge_attr2 = edge_attr.detach().clone().requires_grad_()
    out1, (edges1, alpha1) = conv1(x1, edge_index, edge_attr1,
                                   return_attention_weights=True)
    out2, (edges2, alpha2) = conv2(x2, edge_index, edge_attr2,
                                   return_attention_weights=True)

    assert torch.equal(as_tensor(edges1), as_tensor(edges2))
    torch.testing.assert_close(alpha1, alpha2, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(out1, out2, atol=2e-5, rtol=1e-5)

    (out1.sum() + alpha1.square().sum()).backward()
    (out2.sum() + alpha2.square().sum()).backward()
    assert_parameter_grads_equal(conv1, conv2)
    assert_input_grads_equal(x1, x2)
    torch.testing.assert_close(edge_attr1.grad, edge_attr2.grad, atol=2e-5,
                               rtol=1e-5)
