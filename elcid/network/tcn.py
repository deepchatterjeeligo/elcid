"""Temporal convolutional neural network using PyTorch"""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


_residual_block_default_kwargs = defaultdict(
    stride=1, padding=0, dilation=1, groups=1,
    bias=True, padding_mode='zeros'
)

class ResidualBlock(nn.Module):
    """A residual block as mentioned in Bai et. al. (2018)"""
    def __init__(self, channels, kernel_size,
                 dropout=0, activation=nn.ReLU, **kwargs):
        super().__init__()
        self.channels = channels
        self.activation = activation()
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=kernel_size//2, **kwargs)
        self.conv_one_cross_one = nn.Conv1d(
            channels, channels, 1, **kwargs
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        orig_x = torch.clone(x)  # copy to add later
        orig_x = self.conv_one_cross_one(x)
        for _ in range(2):
            x = self.activation(self.conv(x))
            x = self.dropout(x)
        return x + orig_x


class CausalConv1d(nn.Conv1d):
    """A 1D causal convolution"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dilation, *_ = self.dilation  # assuming only one entry in dilation
        kernel_size, *_ = self.kernel_size  # assuming only one entry in kernel_size
        self.pad = (dilation * (kernel_size - 1), 0)  # only left padding

    def forward(self, x):
        x = F.pad(x, self.pad)
        return super().forward(x)


class TCN(nn.Module):
    def __init__(self, channels=64, kernel_size=3,
                 dilations=(1, 2, 4, 8),
                 dropout=0.):
        super().__init__()
        self.dilations = dilations
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.residual_blocks = list()

    def forward(self, x):
        for dilation in self.dilations:
            # convolve
            conv = CausalConv1d(
                self.channels, self.channels, self.kernel_size,
                dilation=dilation
            )
            x = conv(x)
            # dropout
            x = F.dropout(x, self.dropout)
            # activation
            x = F.softmax(x, dim=1)
            # residual block
            residual_block = ResidualBlock(
                self.channels,
                self.kernel_size,
                dropout=self.dropout,
                dilation=1  # FIXME: Do we need a dilated layer here?
            )
            x = residual_block(x)
        return x