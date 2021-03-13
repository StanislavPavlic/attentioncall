import argparse
import re
from collections import defaultdict

import pytorch_lightning as pl
from edlib import align
import wandb

from datasets import to_seq


split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")


def layers(params_str):
    try:
        return tuple(map(int, params_str.split(',')))
    except:
        raise argparse.ArgumentTypeError(
            "Layer arguments must be in form of out_channels, kernel_size, stride. Example : \'256,10,2\'")


def accuracy(ref, seq):
    alignment = align(seq, ref, mode='HW', task='path')
    counts = defaultdict(int)
    cigar = alignment['cigar']

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    acc = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    
    return acc


class BasecallLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.val_x, self.val_y, self.val_l = val_samples
        self.val_x = self.val_x[:num_samples]
        self.val_l = self.val_l[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_x = self.val_x.to(device=pl_module.device)

        calls = pl_module(val_x)

        table = wandb.Table(columns=["Basecall", "Reference"])

        refs = []
        s = 0
        for l in self.val_l:
            refs.append(to_seq(self.val_y[s:s + l]))
            s += l
        for call, ref in zip(calls, refs):
            table.add_data(call, ref)

        trainer.logger.experiment.log(
            {
                "examples": table,
                "global_step": trainer.global_step
            },
            commit=False
        )
