from argparse import ArgumentParser

import torch

from basecaller import Basecaller
from pore_model import Pore_Model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = ArgumentParser(description='Basecaller')

    # add program level args
    parser.add_argument('--train_set', type=str, help="Path to directory containing training dataset")
    parser.add_argument('--val_set', type=str, help="Path to directory containing validation dataset")

    # add model specific args
    parser = Basecaller.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # init the trainer
    ckpt_callback = ModelCheckpoint(
        monitor='val_ratio',
        save_last=True,
        save_top_k=1,
        filename='basecaller-{epoch:03d}-{val_ratio:.4f}'
    )
    trainer = Trainer.from_argparse_args(args, callbacks=[ckpt_callback], gpus=[1])

    encoder = Pore_Model(args)
    if args.encoder is not None:
        sd = torch.load(args.encoder)
        encoder.load_state_dict(torch.load(args.encoder))

    # init the model with Namespace directly
    model = Basecaller(args, encoder)

    trainer.tune(model)
    trainer.fit(model)
    # trainer.test(model)
