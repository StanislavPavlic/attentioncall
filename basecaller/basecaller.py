import random
import sys
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from fast_ctc_decode import viterbi_search

from Levenshtein import distance, ratio

from datasets import BasecallDataset, pad_collate_fn, base_to_idx, alphabet, to_seq
from pore_model import Pore_Model
from util import layers


class Basecaller(pl.LightningModule):
    def __init__(self, args, encoder: Pore_Model):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma
        self.n_classes = len(base_to_idx)
        self.encoder = encoder
        self.encoder_dim = args.fe_conv_layers[-1][0]
        self.fc = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(self.encoder_dim // 2, self.n_classes)
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if self.args.train_set is None:
            print("Need at least one dataset")
            sys.exit(1)

    def setup(self, stage: str):
        self.train_dataset = BasecallDataset(self.args.train_set, self.args.chunk_size)
        self.val_dataset = BasecallDataset(self.args.val_set, self.args.chunk_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=pad_collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=pad_collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def forward(self, x):
        # x = [T x H]
        # T x H -> T x C
        seqs = []
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = F.softmax(x, 1)
        for x_ in x:
            seq, _ = viterbi_search(x_.cpu().numpy(), alphabet)
            seqs.append(seq)
        return seqs

    def get_loss(self, x, y, l):
        T, N, C = x.shape
        logits = F.log_softmax(x, dim=2)
        logits_lengths = torch.full((N,), T, dtype=torch.int32)
        loss = F.ctc_loss(logits.cpu(), y.cpu(), logits_lengths.cpu(), l.cpu(), zero_infinity=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y, l = batch
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(0, 1)
        loss = self.get_loss(x, y, l)
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, l = batch
        
        x_ = self.encoder(x)
        x_ = x_.transpose(1, 2)
        x_ = self.fc(x_)
        x_ = x_.transpose(0, 1)
        val_loss = self.get_loss(x_, y, l)
        self.log('val_loss', val_loss)
        
        s = 0
        val_distance = 0
        val_ratio = 0
        pred = self.forward(x)
        for i in range(len(l)):
            true = to_seq(y[s:s + l[i]])
            s += l[i]
            val_distance += distance(true, pred[i])            
            val_ratio += ratio(true, pred[i])
        val_distance /= len(l)
        val_ratio /= len(l)

        self.log('val_edit_distance', val_distance)
        self.log('val_ratio', val_ratio)

        return {
            'val_loss': val_loss,
            'val_edit_distance': val_distance,
            'val_ratio': val_ratio
        }

    def test_step(self, batch, batch_idx):
        # x = [N x T], y = [T'], l = [N]
        x, y, l = batch
        # N x T -> N x T x E
        x = self.encoder(x)
        # N x T x E -> N x T x C
        x = self.fc(x)
        # N x T x C -> T x N x C
        x = x.transpose(0, 1)
        loss = self.get_loss(x, y, l)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--chunk_size', type=int, default=512,
                            help="Signal chunk size")

        parser.add_argument('--batch_size', type=int, default=64,
                            help="Size of mini-batch")

        parser.add_argument('--num_workers', type=int, default=4,
                            help="How many subprocesses to use for data loading")

        parser.add_argument('--lr', type=float, default=1e-4,
                            help="Learning rate")

        parser.add_argument('--gamma', type=float, default=1,
                            help="Learning rate decay factor")

        parser.add_argument('--encoder', type=str, default=None,
                            help="Encoder: saved state dictionary.")

        parser.add_argument('--fe_conv_layers', type=layers, nargs='+',
                            default=[(64, 9, 3), (128, 45, 1), (256, 9, 1), (256, 27, 1), (512, 3, 1)],
                            help="Feature encoder: set convolution layers")

        parser.add_argument('--fe_dropout', type=float, default=0.05,
                            help="Feature encoder: dropout")

        parser.add_argument('--fe_bias', default=False, action='store_true',
                            help="Feature encoder: turn on convolution bias")

        parser.add_argument('--fe_residual', default=True, action='store_true',
                            help="Feature encoder: turn on residual connections")

        parser.add_argument('--fe_separable', default=True, action='store_true',
                            help="Feature encoder: turn on separable convolutions")

        parser.add_argument('--fe_repeat', type=int, default=5,
                            help="Feature encoder: number of times a block is repeated, does not apply to first block")

        parser.add_argument('--trns_dim_feedforward', type=int, default=512,
                            help="Transformer: dimension of the feedforward network model used in transformer encoder")

        parser.add_argument('--trns_nhead', type=int, default=8,
                            help="Transformer: number of heads in the multi head attention models")

        parser.add_argument('--trns_n_layers', type=int, default=6,
                            help="Transformer: number of sub-encoder-layers in the transformer encoder")

        parser.add_argument('--trns_dropout', type=float, default=0.05,
                            help="Transformer: dropout")

        parser.add_argument('--trns_activation', type=str, default='gelu',
                            help="Transformer: activation function")

        return parser
