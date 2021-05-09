import math
from typing import List, Tuple

import torch
import torch.nn as nn


class PoreModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.encoder_dim == args.fe_conv_layers[-1][0], "Check that the encoder out dimension fits the number of channels in the last conv layer"
        self.feature_encoder = FeatureEncoder(args.fe_conv_layers, args.fe_bias, args.fe_repeat)
        self.transformer = Transformer(args.encoder_dim, args.trns_nhead, args.trns_dim_feedforward,
                                       args.trns_n_layers, args.trns_dropout, args.trns_activation)

    def forward(self, x, padding_mask=None):
        x = self.feature_encoder(x)  # B x C x T
        x = x.permute(2, 0, 1)  # T x B x C
        x = self.transformer(x, padding_mask=padding_mask)  # T x B x F
        x = x.permute(1, 2, 0)  # B x F x T
        return x


class FeatureEncoderBlock(nn.Module):
    """
    Feature encoder block which consists of a 1D convolution, 1d pooling, batch normalization and a GELU activation.
    """

    def __init__(self, in_channels: int, conv: Tuple[int, int, int], bias: bool = False, repeat: int = 1):
        super(FeatureEncoderBlock, self).__init__()

        out_channels, kernel_size, stride = conv
        self.block = nn.ModuleList()
        for _ in range(repeat):
            self.block.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=bias
                )
            )
            in_channels = out_channels
        if stride > 1:
            self.block.append(
                nn.MaxPool1d(
                    kernel_size=stride,
                    stride=stride
                )
            )
        self.block.extend(
            [
                nn.BatchNorm1d(out_channels),
                nn.GELU()
            ]
        )

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x


class FeatureEncoder(nn.Module):
    """
    Feature encoder which consists of feature encoder blocks that encodes 1D temporal data using convolution.
    """

    def __init__(self, conv_layers: List[Tuple[int, int, int]], bias: bool = False, repeat: int = 1):
        super(FeatureEncoder, self).__init__()

        self.blocks = nn.ModuleList()
        in_channels = 1
        for conv in conv_layers:
            self.blocks.append(
                FeatureEncoderBlock(
                    in_channels=in_channels,
                    conv=conv,
                    bias=bias,
                    repeat=repeat
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
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                               dropout=dropout, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

    def forward(self, x, padding_mask=None):
        # input => S x N x E, S = input sequence time steps, N = batch size, E = features
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return x
