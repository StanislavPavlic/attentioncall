from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from basecaller import Basecaller
from util import BasecallLogger

if __name__ == '__main__':
    wandb_logger = WandbLogger(project="basecaller")

    model_parser = ArgumentParser(add_help=False)

    # add program level args
    model_parser.add_argument('--train_set', type=str, help="Path to directory containing training dataset")
    model_parser.add_argument('--val_set', type=str, help="Path to directory containing validation dataset")

    # add model specific args
    model_parser = Basecaller.add_model_specific_args(model_parser)

    # add all the available trainer options to argparse
    parser = ArgumentParser(description='Basecaller', parents=[model_parser])
    parser = Trainer.add_argparse_args(parser)

    model_args, remaining_args = model_parser.parse_known_args()
    args = parser.parse_args(remaining_args)

    # init the model
    model = Basecaller(model_args)

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
