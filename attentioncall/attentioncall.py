from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Union, Dict
from math import ceil

import pytorch_lightning as pl
import torch
from fast_ctc_decode import viterbi_search, beam_search
from torch import nn
from torch.nn import functional as F

from datasets import base_to_idx, alphabet, to_seq
from poremodel import PoreModel
from util import layers, accuracy, get_cosine_schedule_with_warmup


class FeatureDecoderBlock(nn.Module):
    """
    Feature decoder block which consists of a transposed 1D convolution, batch normalization and a GELU activation.
    """

    def __init__(self, in_channels: int, conv_t: Tuple[int, int, int], bias: bool = False, repeat: int = 1):
        super(FeatureDecoderBlock, self).__init__()

        out_channels, kernel_size, stride = conv_t
        self.block = nn.ModuleList()
        for _ in range(repeat - 1):
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
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=stride,
                    stride=stride,
                    bias=bias
                )
            )
        else:
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


class FeatureDecoder(nn.Module):
    """
    Feature decoder which consists of feature decoder blocks that decodes 1D temporal data using transposed convolution.
    """

    def __init__(self, in_channels: int, conv_t_layers: List[Tuple[int, int, int]], bias: bool = False,
                 repeat: int = 1):
        super(FeatureDecoder, self).__init__()

        self.blocks = nn.ModuleList()
        for conv_t in conv_t_layers:
            self.blocks.append(
                FeatureDecoderBlock(
                    in_channels=in_channels,
                    conv_t=conv_t,
                    bias=bias,
                    repeat=repeat
                )
            )
            in_channels = conv_t[0]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AttentionCall(pl.LightningModule):
    def __init__(self, args: Union[Namespace, Dict], train_mode=True):
        super().__init__()

        if train_mode:
            self.save_hyperparameters(args)
        else:
            args = Namespace(**args)

        for k, v in vars(args).items():
            setattr(self, k, v)

        encoder = PoreModel(args)
        if self.encoder is not None:
            encoder.load_state_dict(torch.load(self.encoder))
        self.encoder = encoder
        self.decoder = FeatureDecoder(args.encoder_dim, args.fd_conv_t_layers, args.fd_bias, args.fd_repeat)
        self.n_classes = len(base_to_idx)
        self.fc = nn.Linear(args.fd_conv_t_layers[-1][0], self.n_classes)
        self.downsample_factor = 1
        for conv in args.fe_conv_layers:
            self.downsample_factor *= conv[2]
        self.upsample_factor = 1
        for conv_t in args.fd_conv_t_layers:
            self.upsample_factor *= conv_t[2]

    def forward(self, x, beam_size=1, pad=None):
        with torch.no_grad():
            seqs = []
            if pad is not None:
                batch_size = x.shape[0]
                padding_mask = torch.zeros(
                    (batch_size, self.chunk_size // self.downsample_factor)
                ).type(torch.BoolTensor).to(x.device)
                pad = ceil(pad / self.downsample_factor)
                padding_mask[-1, -pad:] = True
            else:
                padding_mask = None
            x = self.encoder(x, padding_mask=padding_mask)
            x = self.decoder(x)
            x = x.transpose(1, 2)
            x = self.fc(x)
            if pad is not None:
                pad *= self.upsample_factor
                x[-1, -pad:, 1:] -= float('inf')
            x = F.softmax(x, dim=-1)
            for x_ in x:
                if beam_size == 1:
                    seq, _ = viterbi_search(x_.cpu().numpy(), alphabet)
                else:
                    seq, _ = beam_search(x_.cpu().numpy(), alphabet, beam_size=beam_size, beam_cut_threshold=0)
                seqs.append(seq)
        return seqs

    def get_loss(self, x, y, l):
        T, N, C = x.shape
        logits = F.log_softmax(x, dim=2)
        logits_lengths = torch.full((N,), T, dtype=torch.int32, device='cpu')
        loss = F.ctc_loss(logits, y.cpu(), logits_lengths, l.cpu(), zero_infinity=False)
        return loss

    def training_step(self, batch, batch_idx):
        x, y, l = batch
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(0, 1)
        loss = self.get_loss(x, y, l)
        if torch.isfinite(loss):
            self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, l = batch

        x_ = self.encoder(x)
        x_ = self.decoder(x_)
        x_ = x_.transpose(1, 2)
        x_ = self.fc(x_)
        x_ = x_.transpose(0, 1)
        val_loss = self.get_loss(x_, y, l)
        if torch.isfinite(val_loss):
            self.log('val/loss', val_loss, sync_dist=True)

        s = 0
        val_acc = 0
        pred = self.forward(x)
        n = len(l)
        for i in range(len(l)):
            true = to_seq(y[s:s + l[i]])
            s += l[i]
            acc = accuracy(true, pred[i])
            if acc is None:
                n -= 1
            else:
                val_acc += acc
        if n > 0:
            val_acc /= n

        self.log('val/accuracy', val_acc, sync_dist=True)

    def configure_optimizers(self):
        optimizers = [torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=self.gamma)
        ]
        return optimizers, schedulers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--chunk_size', type=int, default=1024,
                            help="Signal chunk size")

        parser.add_argument('--batch_size', type=int, default=128,
                            help="Size of mini-batch")

        parser.add_argument('--num_workers', type=int, default=4,
                            help="How many subprocesses to use for data loading")

        parser.add_argument('--lr', type=float, default=1e-4,
                            help="Learning rate")

        parser.add_argument('--gamma', type=float, default=0.95,
                            help="Learning rate decay factor")

        parser.add_argument('--encoder', type=str, default=None,
                            help="Encoder: saved state dictionary")

        parser.add_argument('--encoder_dim', type=int, default=512,
                            help="Dimension of encoder output")

        parser.add_argument('--fe_conv_layers', type=layers, nargs='+',
                            default=[
                                (64, 3, 2),
                                (128, 3, 2),
                                (256, 3, 2),
                                (512, 3, 1)
                            ],
                            help="Feature encoder: set convolution layers")

        parser.add_argument('--fe_bias', default=False, action='store_true',
                            help="Feature encoder: turn on convolution bias")

        parser.add_argument('--fe_repeat', type=int, default=1,
                            help="Feature encoder: number of times a block is repeated")

        parser.add_argument('--trns_dim_feedforward', type=int, default=2048,
                            help="Transformer: dimension of the feedforward network model used in transformer encoder")

        parser.add_argument('--trns_nhead', type=int, default=8,
                            help="Transformer: number of heads in the multi head attention models")

        parser.add_argument('--trns_n_layers', type=int, default=10,
                            help="Transformer: number of sub-encoder-layers in the transformer encoder")

        parser.add_argument('--trns_dropout', type=float, default=0.05,
                            help="Transformer: dropout")

        parser.add_argument('--trns_activation', type=str, default='gelu',
                            help="Transformer: activation function")

        parser.add_argument('--fd_conv_t_layers', type=layers, nargs='+',
                            default=[
                                (512, 3, 1),
                                (256, 3, 2),
                                (256, 3, 1)
                            ],
                            help="Feature decoder: set convolution transpose layers")

        parser.add_argument('--fd_bias', default=False, action='store_true',
                            help="Feature decoder: turn on convolution bias")

        parser.add_argument('--fd_repeat', type=int, default=1,
                            help="Feature decoder: number of times a block is repeated")

        return parser
