import argparse
import re
from collections import defaultdict

import pytorch_lightning as pl
from edlib import align, getNiceAlignment
import wandb

from datasets import to_seq


split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")


def layers(params_str):
    try:
        return tuple(map(int, params_str.split(',')))
    except:
        raise argparse.ArgumentTypeError(
            "Layer arguments must be in form of out_channels, kernel_size, stride. Example : \'256,10,2\'")


def accuracy(ref, seq, nice=False):
    alignment = align(seq, ref, mode='HW', task='path')
    counts = defaultdict(int)
    cigar = alignment['cigar']

    if cigar is None:
        if nice:
            return None, None
        return None

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    acc = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    
    if nice:
        nice_align = getNiceAlignment(alignment, seq, ref)
        nice_str = '\n'.join([nice_align['query_aligned'], nice_align['matched_aligned'], nice_align['target_aligned']])
        return acc, nice_str
    return acc


class BasecallLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=10):
        super().__init__()
        self.val_samples = val_samples
        if val_samples is None:
            return
        self.val_x, self.val_y, self.val_l = val_samples
        self.val_x = self.val_x[:num_samples]
        self.val_l = self.val_l[:num_samples]
        self.refs = []
        s = 0
        for l in self.val_l:
            self.refs.append(to_seq(self.val_y[s:s + l]))
            s += l

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_samples is None:
            return
        val_x = self.val_x.to(device=pl_module.device)

        calls = pl_module(val_x)

        table = wandb.Table(columns=["Basecall", "Reference", "Identity", "Alignment"])

        for call, ref in zip(calls, self.refs):
            acc, alignment = accuracy(call, ref, nice=True)
            if acc is not None:
                table.add_data(call, ref, acc, alignment)

        trainer.logger.experiment.log(
            {
                "examples": table,
                "global_step": trainer.global_step
            },
            commit=False
        )
