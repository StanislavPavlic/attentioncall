import sys
import time
from pathlib import Path

import numpy as np
import torch
from ont_fast5_api.fast5_interface import get_fast5_file
from tqdm import tqdm

from basecaller import Basecaller


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
    for file in tqdm(files):
        with get_fast5_file(str(file), mode='r') as f5:
            reads += [(read.read_id, torch.from_numpy(get_signal_data(read))) for read in f5.get_reads()]
    return reads


def load_model(path):
    model = Basecaller.load_from_checkpoint(path, train_mode=False)
    return model


def basecall(model, reads, batch_size=32, chunk_size=0, beamsize=1):
    for read_id, read in reads:
        pass


if __name__ == '__main__':
    start_time = time.time()

    with open("basecalls.fasta", "w") as f:
        f.write("")

    model_path = sys.argv[1]
    read_path = sys.argv[2]
    model = load_model(model_path)
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch_size = model.batch_size
    chunk_size = model.chunk_size
    reads = get_reads(read_path)
    for i, (read_id, read) in enumerate(tqdm(reads)):
        T = read.shape[0]
        regular_remainder = T % (batch_size * chunk_size)
        full_chunk_remainder = regular_remainder % chunk_size
        batches = list(read[:-regular_remainder].view(-1, batch_size, chunk_size))
        incomplete_batch = read[-regular_remainder:-full_chunk_remainder if full_chunk_remainder else None].view(-1,
                                                                                                                 chunk_size)
        if len(incomplete_batch) > 0:
            batches.append(incomplete_batch)
        chunk_remainder = read[-full_chunk_remainder:].view(1, -1)
        if full_chunk_remainder > 0:
            batches.append(chunk_remainder)

        basecalled_seq = ""
        for batch in batches:
            batch = batch.to(device)
            pred = model(batch, beam_size=5)
            basecalled_seq += ''.join(pred)

        with open("basecalls.fasta", "a") as f:
            f.writelines([">" + read_id + "\n", basecalled_seq + "\n"])
    print(f"Seconds elapsed: {time.time() - start_time} s")
