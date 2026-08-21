import pytest
import torch

from torch_geometric import EdgeIndex
from torch_geometric.nn import CuGraphSAGEConv, SAGEConv
from torch_geometric.testing import onlyCUDA, withPackage


@onlyCUDA
@withPackage('cudnn')
@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max'])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('bipartite', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('project', [True, False])
@pytest.mark.parametrize('root_weight', [True, False])
def test_sage_conv_equality(aggr, bias, bipartite, normalize, project,
                            root_weight):
    in_channels = (8, 4) if bipartite else 8
    out_channels = 16
    kwargs = dict(aggr=aggr, bias=bias, normalize=normalize, project=project,
                  root_weight=root_weight)

    x_src1 = torch.rand(10, 8, device='cuda', requires_grad=True)
    x_src2 = x_src1.detach().clone().requires_grad_(True)
    if bipartite:
        x_dst1 = torch.rand(8, 4, device='cuda', requires_grad=True)
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

    conv1 = SAGEConv(in_channels, out_channels, **kwargs).cuda()
    conv2 = CuGraphSAGEConv(in_channels, out_channels, **kwargs).cuda()
    conv2.load_state_dict(conv1.state_dict())

    out1 = conv1(x1, edge_index, size=size)
    out2 = conv2(x2, edge_index, size=size)
    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    assert torch.allclose(x_src1.grad, x_src2.grad, atol=1e-6)
    if bipartite:
        if root_weight:
            assert torch.allclose(x_dst1.grad, x_dst2.grad, atol=1e-6)
        else:
            assert x_dst1.grad is None
            assert x_dst2.grad is None

    params1 = dict(conv1.named_parameters())
    params2 = dict(conv2.named_parameters())
    assert params1.keys() == params2.keys()
    for name in params1.keys():
        assert torch.allclose(params1[name].grad, params2[name].grad,
                              atol=1e-6)


@onlyCUDA
@withPackage('cudnn')
def test_sage_conv_without_target_features():
    x1 = torch.rand(4, 8, device='cuda', requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device='cuda')

    conv1 = SAGEConv((8, 16), 32).cuda()
    conv2 = CuGraphSAGEConv((8, 16), 32).cuda()
    conv2.load_state_dict(conv1.state_dict())

    out1 = conv1((x1, None), edge_index, size=(4, 2))
    out2 = conv2((x2, None), edge_index, size=(4, 2))
    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)


@onlyCUDA
@withPackage('cudnn')
def test_lazy_sage_conv():
    x = torch.rand(4, 8, device='cuda')
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device='cuda')

    conv = CuGraphSAGEConv(-1, 32).cuda()
    out = conv(x, edge_index)
    assert out.size() == (4, 32)

    with pytest.raises(ValueError, match="does not support lazy"):
        CuGraphSAGEConv(-1, 32, project=True)


@onlyCUDA
@withPackage('cudnn')
def test_sage_conv_edge_index():
    x = torch.rand(4, 8, device='cuda')
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device='cuda')
    csc = EdgeIndex(edge_index, sparse_size=(4, 2), sort_order='col')
    conv = CuGraphSAGEConv((8, 8), 32).cuda()

    out1 = conv((x, x[:2]), edge_index, size=(4, 2))
    out2 = conv((x, x[:2]), csc)
    assert torch.allclose(out1, out2, atol=1e-6)


@onlyCUDA
@withPackage('cudnn')
def test_sage_conv_invalid_aggr():
    with pytest.raises(ValueError, match="Aggregation function"):
        CuGraphSAGEConv(8, 32, aggr='lstm')
