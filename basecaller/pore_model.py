import math
from typing import List, Tuple

import torch
import torch.nn as nn


class Pore_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_encoder = FeatureEncoder(args.fe_conv_layers, args.fe_bias, args.fe_residual, args.fe_dropout,
                                              args.fe_repeat, args.fe_separable)
        self.transformer = Transformer(args.fe_conv_layers[-1][0], args.trns_nhead, args.trns_dim_feedforward,
                                       args.trns_n_layers, args.trns_dropout, args.trns_activation)

    def forward(self, x):
        # T is generic
        # BN x C x T
        x = self.feature_encoder(x)
        # T x BN x C
        x = x.permute(2, 0, 1)
        # T x BN x F
        x = self.transformer(x)
        # BN x F x T
        x = x.permute(1, 2, 0)
        return x


class FeatureEncoderBlock(nn.Module):
    """
    Feature encoder block which consists of a 1D convolution, batch normalization and a ReLU activation.
    """

    def __init__(self, in_channels: int, conv: Tuple[int, int, int], bias: bool = False, dropout: float = 0.0,
                 separable: bool = False, repeat: int = 1, residual: bool = False):
        super(FeatureEncoderBlock, self).__init__()

        self.use_residual = residual

        self.block = nn.ModuleList()
        out_channels, conv_kernel_size, conv_stride = conv
        in_channels_ = in_channels
        for _ in range(repeat - 1):
            self.block.extend(
                self.get_elem(in_channels_, out_channels, conv_kernel_size, conv_stride, bias, separable)
            )
            in_channels_ = out_channels
        self.block.extend(
            self.get_elem(in_channels_, out_channels, conv_kernel_size, conv_stride, bias, separable)
        )

        self.block.append(
            nn.BatchNorm1d(out_channels)
        )

        if residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1,
                    padding=0, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

        self.activation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_ = x
        for layer in self.block:
            x = layer(x)
        if self.use_residual:
            x = x + self.residual(x_)
        x = self.activation(x)
        return x

    @staticmethod
    def get_elem(in_channels, out_channels, conv_kernel_size, conv_stride, bias, separable):
        if separable:
            elem = [
                nn.Conv1d(
                    in_channels, in_channels, kernel_size=conv_kernel_size, stride=conv_stride,
                    padding=conv_kernel_size // 2, bias=bias, groups=in_channels
                ),
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1,
                    padding=0, bias=bias
                )
            ]
        else:
            elem = [
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride,
                    padding=conv_kernel_size // 2, bias=bias
                )
            ]

        return elem


class FeatureEncoder(nn.Module):
    """
    Feature encoder which consists of feature encoder blocks that encodes 1D temporal data using convolution.
    """

    def __init__(self, conv_layers: List[Tuple[int, int, int]], bias: bool = False, residual: bool = False,
                 dropout: float = 0.0, repeat: int = 1, separable: bool = True):
        super(FeatureEncoder, self).__init__()

        self.residual = residual

        self.blocks = nn.ModuleList()
        in_channels = 1
        self.blocks.append(
            FeatureEncoderBlock(
                in_channels=in_channels,
                conv=conv_layers[0],
                bias=bias,
                dropout=dropout,
                separable=False,
                repeat=1,
                residual=False
            )
        )
        in_channels = conv_layers[0][0]
        for conv in conv_layers[1:]:
            self.blocks.append(
                FeatureEncoderBlock(
                    in_channels=in_channels,
                    conv=conv,
                    bias=bias,
                    dropout=dropout,
                    separable=separable,
                    repeat=repeat,
                    residual=residual
                )
            )
            in_channels = conv[0]

    def forward(self, x):
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, nhead, dim_feedforward=1024, n_layers=8, dropout=0.1, activation='gelu'):
        super(Transformer, self).__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=dim_feedforward,
                                               nhead=nhead, activation=activation, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

    def forward(self, x, mask=None):
        # input => S x N x E, S = input sequence time steps, N = batch size, E = features
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask)
        return x
