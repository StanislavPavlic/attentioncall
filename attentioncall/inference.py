import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ont_fast5_api.fast5_interface import get_fast5_file
from tqdm import tqdm

from attentioncall import AttentionCall


def get_files(path, recursive=False):
    if path.is_file():
        return [path]
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')
    return list(files)


def med_mad(data, factor=1.4826):
    med = np.median(data)
    mad = factor * np.median(np.absolute(data - med))
    return med, mad


def get_signal_data(read):
    signal_data = read.get_raw_data(scale=True)
    med, mad = med_mad(signal_data)
    signal_data = (signal_data - med) / mad  # normalise
    return signal_data


def get_reads(path, recursive=False):
    files = get_files(Path(path), recursive)
    reads = []
    for file in tqdm(files, desc='loading reads'):
        with get_fast5_file(str(file), mode='r') as f5:
            reads += [(read.read_id, get_signal_data(read)) for read in f5.get_reads()]
    return reads


def load_model(path):
    model = AttentionCall.load_from_checkpoint(path, train_mode=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AttentionCall inference runner')
    parser.add_argument('model', help='path to model checkpoint file .ckpt')
    parser.add_argument('reads', help='path to raw signal reads in .fast5 format')
    parser.add_argument('out', help='path to .fasta output file')
    parser.add_argument('--device', default='cpu',
                        help='device to run the inference on, one of:'
                             '[\'cpu\', \'cuda\', \'cuda:X\'] where X is the device ordinal')
    args = parser.parse_args()
    model_path = args.model
    read_path = args.reads
    out_path = args.out
    device = args.device

    start_time = time.time()

    with open(out_path, "w") as f:
        f.write("")

    model = load_model(model_path)
    model.eval()
    if device.startswith('cuda'):
        assert torch.cuda.is_available(), 'cuda is not available'
    device = torch.device(device)
    model.to(device)
    batch_size = model.batch_size
    chunk_size = model.chunk_size

    reads = get_reads(read_path)
    for i, (read_id, read) in enumerate(tqdm(reads, desc='basecalling reads')):
        T = read.shape[0]
        remainder = T % (batch_size * chunk_size)
        last_batch_size = remainder // chunk_size + 1
        batches = list(read[:-remainder].reshape(-1, batch_size, chunk_size))
        pads = [None] * len(batches)

        if remainder > 0:
            last_batch = np.zeros(last_batch_size * chunk_size, dtype=np.float32)
            last_batch[:remainder] = read[-remainder:]
            last_batch = last_batch.reshape(last_batch_size, chunk_size)
            pad = last_batch_size * chunk_size - remainder
            batches.append(last_batch)
            pads.append(pad)

        basecalled_seq = ""
        for batch, pad in zip(batches, pads):
            batch = torch.from_numpy(batch).to(device)
            pred = model(batch, pad=pad, beam_size=1)
            basecalled_seq += ''.join(pred)

        with open(out_path, "a") as f:
            f.writelines([">" + read_id + "\n", basecalled_seq + "\n"])

    print(f"seconds elapsed: {time.time() - start_time} s")
    print("basecalling done.")
    print(f"sequences basecalled: {len(reads)}")
