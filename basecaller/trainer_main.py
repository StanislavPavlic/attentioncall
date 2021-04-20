from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from basecaller import Basecaller
from datasets import BasecallDataModule
from util import BasecallLogger

if __name__ == '__main__':
    seed_everything(seed=42)
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

    # init the datamodule
    datamodule = BasecallDataModule(
        model_args.train_set,
        model_args.val_set,
        model_args.batch_size,
        model_args.chunk_size,
        model_args.num_workers
    )
    #datamodule.prepare_data()
    #datamodule.setup()
    # get samples for logging
    #samples = next(iter(datamodule.val_dataloader()))
    samples = None
    # init the trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        gpus=[1],
        auto_select_gpus=False,
        callbacks=[BasecallLogger(samples)]
    )

    trainer.tune(
        model=model,
        datamodule=datamodule
    )
    trainer.fit(
        model=model,
        datamodule=datamodule
    )
    # trainer.test(model)
