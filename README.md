# AttentionCall
AttentionCall is a Python implementation of a deep learning model for basecalling DNA nanopore sequencing reads.
The implementation is based on PyTorch and PyTorch Lightning. It was developed as part of my Master's thesis.

AttentionCall can be used to train new models or run inference on your data.

## Installation
To install AttentionCall from source:
```bash
# for HTTPS
git clone https://github.com/StanislavPavlic/attentioncall.git
# for SSH
git clone git@github.com:StanislavPavlic/attentioncall.git
# go to main directory
cd attentioncall
# create a virtual environment
python3 -m venv venv
# install the requirements
pip install -r requirements.txt
```

## Usage
AttentionCall uses [WandB][wandb] for experiment tracking. If you wish to use something else, you can change it in [trainer_main.py][trainer].
Before running anything, please refer to the help message:
```bash
python trainer_main.py --help
```

For training a model run with default parameters:
```bash
python trainer_main.py --train_set /path/to/train_set.hdf5 --val_set /path/to/validation_set.hdf5
```
If you wish to define your parameters, see the help message for a list of available parameters.

Before running inference, please refer to the help message:
```bash
python inference.py --help
```

For running inference with a model checkpoint:
```bash
python inference.py /path/to/model_checkpoint.ckpt /path/to/read_directory /path/to/output_file.fasta --device cuda
```

## Data format
Inference works with standard FAST5 files.

Training uses HDF5 files in a format defined by [Taiyaki][taiyaki].

## Disclaimer
Laboratory for Bioinformatics and Computational Biology cannot be held responsible for any copyright infringement caused by actions of students contributing to any of its repositories. Any case of copyright infringement will be promptly removed from the affected repositories and reported to appropriate faculty organs.

[wandb]: wandb.ai
[trainer]: github.com/StanislavPavlic/attentioncall/blob/main/attentioncall/trainer_main.py
[taiyaki]: github.com/nanoporetech/taiyaki/blob/master/docs/FILE_FORMATS.md#mapped-signal-files-v-8