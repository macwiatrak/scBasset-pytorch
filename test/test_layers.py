import torch

import torch.nn.functional as F

from scbasset.layers import ConvLayer, DenseLayer


def test_conv_layer_forward():
    """Test forward pass of scBasset conv layer module."""
    # create dummy data
    batch_size = 8
    seq_length = 128
    in_channels = 4
    out_channels = 18
    kernel_size = 17
    pool_size = 3
    x = (
        F.one_hot(
            torch.randint(0, in_channels, (batch_size, seq_length)),
            num_classes=in_channels,
        )
        .transpose(-2, -1)
        .type(torch.float32)
    )

    conv_layer = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        pool_size=pool_size,
        batch_norm=True,
    )
    out = conv_layer(x)
    assert out.shape == (batch_size, out_channels, seq_length // pool_size)


def test_dense_layer_forward():
    """Test forward pass of scBasset dense layer module."""
    # create dummy data
    batch_size = 8
    in_features = 128
    out_features = 18
    x = torch.rand(batch_size, in_features)

    dense_layer = DenseLayer(
        in_features=in_features,
        out_features=out_features,
        batch_norm=True,
    )
    out = dense_layer(x)
    assert out.shape == (batch_size, out_features)
