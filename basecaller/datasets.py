import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pytorch_lightning as pl

base_to_idx = {
    'N': 0,
    'A': 1,
    'C': 2,
    'G': 3,
    'T': 4
}

idx_to_base = {
    0: 'N',
    1: 'A',
    2: 'C',
    3: 'G',
    4: 'T'
}

alphabet = "NACGT"


def to_seq(idxs: torch.Tensor):
    seq = ""
    for idx in idxs:
        seq += idx_to_base[idx.item()]
    return seq


def chunkify(signal, ref_to_signal, reference, chunk_len=512):
    sig_len = len(signal)
    ref_len = len(reference)

    examples = []

    map_s = 0
    while ref_to_signal[map_s] == -1:
        map_s += 1
    map_e = ref_len
    while ref_to_signal[map_e] == sig_len:
        map_e -= 1

    ref_s = map_s
    chunk_s = ref_to_signal[map_s]
    for i in range(map_s, map_e + 1):
        if ref_to_signal[i] - chunk_s >= chunk_len:
            ref_e = i
            chunk_e = chunk_s + chunk_len

            examples.append((signal[chunk_s:chunk_e], reference[ref_s:ref_e]))

            ref_s = ref_e
            chunk_s = ref_to_signal[i]

    return examples


def med_mad(data, factor=1.4826):
    med = np.median(data)
    mad = factor * np.median(np.absolute(data - med))

    return med, mad


def discrete_to_continuous(signal, attributes):
    range_val = attributes['range']
    digitisation = attributes['digitisation']
    offset = attributes['offset']

    scale = range_val / digitisation
    signal = scale * (signal + offset)

    med, mad = med_mad(signal)
    signal = (signal - med) / mad

    return signal


def adjust_reference(reference, attributes):
    alphabet = attributes['alphabet']
    for i, base in enumerate(reference):
        reference[i] = base_to_idx[alphabet[base]]
    return reference


class BasecallDataset(Dataset):
    def __init__(self, path: Path, chunk_len=512):
        super(BasecallDataset, self).__init__()

        path = Path(path)
        assert path.is_file()

        self.examples = []
        with h5py.File(path, mode="r") as h5:
            for read_group in tqdm(h5['Reads'].values(), desc="Loading dataset"):
                attributes = dict(read_group.attrs.items())
                signal = np.array(read_group['Dacs'], dtype=np.float32)
                ref_to_signal = np.array(read_group['Ref_to_signal'])
                reference = np.array(read_group['Reference'], dtype=np.int32)

                signal = discrete_to_continuous(signal, attributes)
                reference = adjust_reference(reference, attributes)

                self.examples.extend(
                    chunkify(signal, ref_to_signal, reference, chunk_len)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        signal, reference = self.examples[idx]
        signal = torch.from_numpy(signal)
        reference = torch.from_numpy(reference)

        return signal, reference, len(reference)


def pad_collate_fn(batch):
    batch = list(zip(*batch))
    elem = batch[0][0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch[0]])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    batch[0] = torch.stack(batch[0], dim=0, out=out)
    batch[1] = torch.cat(batch[1], dim=0)
    batch[2] = torch.tensor(batch[2], dtype=torch.int32)

    return batch


class BasecallDataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, batch_size=128, chunk_size=512, num_workers=0):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.num_workers = num_workers

    def prepare_data(self):
        if self.train_set is None:
            print("Need at least one dataset")
            sys.exit(1)

    def setup(self, stage: str = None):
        self.train_dataset = BasecallDataset(self.train_set, self.chunk_size)
        self.val_dataset = BasecallDataset(self.val_set, self.chunk_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=pad_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=pad_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path)
    parser.add_argument('--chunk_len', type=int, default=512)

    return parser.parse_args()


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    args = get_args()

    ds = BasecallDataset(args.data_path, args.chunk_len)
    # print(ds[84])
    #
    # print(ds[12])
    #
    # print(ds[101])

    for b in DataLoader(ds, collate_fn=pad_collate_fn, batch_size=4):
        x, y, l = b
        print(x, y, l)
