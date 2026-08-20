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
@pytest.mark.parametrize('max_num_neighbors', [8, None])
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('project', [True, False])
@pytest.mark.parametrize('root_weight', [True, False])
def test_sage_conv_equality(aggr, bias, bipartite, max_num_neighbors,
                            normalize, project, root_weight):

    in_channels, out_channels = (8, 16)
    kwargs = dict(aggr=aggr, bias=bias, normalize=normalize, project=project,
                  root_weight=root_weight)

    size = (10, 8) if bipartite else (10, 10)
    x1 = torch.rand(size[0], in_channels, device='cuda', requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    edge_index = torch.tensor([
        [7, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ], device='cuda')

    conv1 = SAGEConv(in_channels, out_channels, **kwargs).cuda()
    conv2 = CuGraphSAGEConv(in_channels, out_channels, **kwargs).cuda()

    with torch.no_grad():
        if project:
            conv2.pre_lin.weight.copy_(conv1.lin.weight)
            conv2.pre_lin.bias.copy_(conv1.lin.bias)
        conv2.lin.weight.data[:, :in_channels] = conv1.lin_l.weight.data
        if root_weight:
            conv2.lin.weight.data[:, in_channels:] = conv1.lin_r.weight.data
        if bias:
            conv2.lin.bias.data[:] = conv1.lin_l.bias.data

    if bipartite:
        out1 = conv1((x1, x1[:size[1]]), edge_index)
    else:
        out1 = conv1(x1, edge_index)

    out2 = conv2(
        x2,
        EdgeIndex(edge_index, sparse_size=size),
        max_num_neighbors=max_num_neighbors,
    )
    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)

    if project:
        assert torch.allclose(
            conv1.lin.weight.grad,
            conv2.pre_lin.weight.grad,
            atol=1e-6,
        )
        assert torch.allclose(
            conv1.lin.bias.grad,
            conv2.pre_lin.bias.grad,
            atol=1e-6,
        )

    assert torch.allclose(
        conv1.lin_l.weight.grad,
        conv2.lin.weight.grad[:, :in_channels],
        atol=1e-6,
    )

    if root_weight:
        assert torch.allclose(
            conv1.lin_r.weight.grad,
            conv2.lin.weight.grad[:, in_channels:],
            atol=1e-6,
        )

    if bias:
        assert torch.allclose(
            conv1.lin_l.bias.grad,
            conv2.lin.bias.grad,
            atol=1e-6,
        )
