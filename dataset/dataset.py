import copy
import torch
from torch.utils.data import Dataset as TorchDataset

from .datareader import DataReader
from .embedding import Embeddings
from ..utils import pad_dial


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


class WordToInt(object):

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def __call__(self, sample):
        for t in sample.turns:
            t.user = [self.embeddings.w2id[tk] for tk in t.user]
            t.system = [self.embeddings.w2id[tk] for tk in t.system]
        return sample


class Padding(object):

    def __init__(self, pad_token: int, max_dial_len: int, max_turn_len: int):
        self.max_dial_len = max_dial_len
        self.max_turn_len = max_turn_len
        self.pad_token = pad_token

    def __call__(self, sample):
        user_turns = [t.user for t in sample.turns]
        return pad_dial(user_turns, self.max_dial_len, self.max_turn_len, self.pad_token)


class ToTensor(object):

    def __call__(self, sample):
        return {'dialogue': sample}
