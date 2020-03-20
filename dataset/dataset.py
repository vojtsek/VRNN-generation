import copy
import torch
from torch.utils.data import Dataset as TorchDataset

from .datareader import DataReader


class Dataset(TorchDataset):
    def __init__(self, reader: DataReader, transform=None):
        self.reader = reader
        self.transform = transform

    def __len__(self):
        return self.reader.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.reader.dialogues[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):

    def __call__(self, sample):
        return {'dialogue': sample}