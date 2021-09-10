import pytest
import torch

from elcid import __version__

from elcid.network import tcn

def test_version():
    assert __version__ == '0.1.0'


@pytest.mark.parametrize(
    'channels,kernel_size,dilation,shape,first,last',
    [[(2, 1), 3, 1, (1, 2, 5), 2, 6],
     [(3, 1), 3, 1, (1, 3, 10), 3, 9],
     [(3, 1), 3, 2, (1, 3, 10), 3, 9]]
)
def test_causal_conv(channels, kernel_size,
                     dilation, shape, first, last):
    causal_conv = tcn.CausalConv1d(
        *channels, kernel_size, dilation=dilation
    )
    # set weights to unity, bias to zero
    causal_conv.weight.data.fill_(1.0)
    causal_conv.bias.data.fill_(0.0)

    X = torch.ones(*shape)
    res = causal_conv(X)
    assert res[0][0][0] == first
    assert res[0][0][-1] == last

@pytest.mark.parametrize(
    'channels,kernel_size,dilation,input_shape',
    [[2, 1, (1,), (1, 2, 5)],
     [3, 3, (1,), (1, 3, 15)],
     [3, 3, (1, 2), (1, 3, 15)],
     [6, 5, (1, 2, 4), (1, 6, 50)]]
)
def test_tcn_basic(channels, kernel_size, dilation, input_shape):
    """Test arbitrary input shape/hyperparam combinations"""
    input = torch.randn(*input_shape)
    net = tcn.TCN(
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilation    
    )
    res = net(input)
    assert res.shape == input.shape