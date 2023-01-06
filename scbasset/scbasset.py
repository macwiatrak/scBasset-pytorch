import torch
import torch.nn as nn

from scbasset.layers import ConvLayer, DenseLayer
from scbasset.utils import _round


class scBasset(nn.Module):
    def __init__(
        self,
        n_cells: int,
        n_filters_init: int = 288,
        n_repeat_blocks_tower: int = 6,
        filters_mult: float = 1.122,
        n_filters_pre_bottleneck: int = 256,
        n_bottleneck_layer: int = 32,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.stem = ConvLayer(
            in_channels=4,
            out_channels=n_filters_init,
            kernel_size=17,
            pool_size=3,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        tower_layers = []
        curr_n_filters = n_filters_init
        for i in range(n_repeat_blocks_tower):
            tower_layers.append(
                ConvLayer(
                    in_channels=curr_n_filters,
                    out_channels=_round(curr_n_filters * filters_mult),
                    kernel_size=5,
                    pool_size=2,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
            )
            curr_n_filters = _round(curr_n_filters * filters_mult)
        self.tower = nn.Sequential(*tower_layers)

        self.pre_bottleneck = ConvLayer(
            in_channels=curr_n_filters,
            out_channels=n_filters_pre_bottleneck,
            kernel_size=1,
            dropout=dropout,
            batch_norm=batch_norm,
            pool_size=2,
        )
        self.bottleneck = DenseLayer(
            in_features=n_filters_pre_bottleneck * 7,
            out_features=n_bottleneck_layer,
            use_bias=True,
            batch_norm=False,
            dropout=0.2,
            activation_fn=nn.Identity(),
        )
        self.final = nn.Linear(n_bottleneck_layer, n_cells)

    def forward(
        self,
        x: torch.Tensor,  # input shape: (batch_size, 4, seq_length)
    ):
        # TODO: add random shift to act as a regularizer on the dataset level
        # TODO: add use reverse complement randomly on the dataset level
        x = self.stem(x)
        x = self.tower(x)
        x = self.pre_bottleneck(x)
        # flatten the input
        x = x.view(x.shape[0], -1)
        x = self.bottleneck(x)
        x = self.final(x)
        return x
