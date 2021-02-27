from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from basecaller import Basecaller
from poremodel import PoreModel
from util import BasecallLogger

if __name__ == '__main__':
    wandb_logger = WandbLogger(project="basecaller")
    parser = ArgumentParser(description='Basecaller')

    # add program level args
    parser.add_argument('--train_set', type=str, help="Path to directory containing training dataset")
    parser.add_argument('--val_set', type=str, help="Path to directory containing validation dataset")

    # add model specific args
    parser = Basecaller.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # init the model
    encoder = PoreModel(args)
    if args.encoder is not None:
        sd = torch.load(args.encoder)
        encoder.load_state_dict(torch.load(args.encoder))
    model = Basecaller(args, encoder)

    # get samples for logging
    samples = next(iter(model.val_dataloader()))
    # init the trainer
    trainer = Trainer.from_argparse_args(
        args,
        gpus=[1],
        callbacks=[BasecallLogger(samples)]
    )

    trainer.tune(model)
    trainer.fit(model)
    # trainer.test(model)
