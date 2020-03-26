import copy
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset

from .embedding import Embeddings
from ..utils import pad_dial


class Dataset(TorchDataset):
    def __init__(self, dialogues: [], transform=None):
        self.dialogues = dialogues
        self.transform = transform

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = copy.deepcopy(self.dialogues[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class WordToInt(object):

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def __call__(self, sample):
        for t in sample.turns:
            t.user = [self.embeddings.w2id[tk] for tk in t.user + [Embeddings.EOS]]
            t.system = [self.embeddings.w2id[tk] for tk in t.system + [Embeddings.EOS]]
        return sample


class Padding(object):

    def __init__(self, pad_token: int, max_dial_len: int, max_turn_len: int):
        self.max_dial_len = max_dial_len
        self.max_turn_len = max_turn_len
        self.pad_token = pad_token

    def __call__(self, sample):
        user_turns = [t.user for t in sample.turns]
        system_turns = [t.system for t in sample.turns]
        user_padded_dial, user_turn_lens = pad_dial(
            user_turns, self.max_dial_len, self.max_turn_len, self.pad_token)
        system_padded_dial, system_turn_lens = pad_dial(
            system_turns, self.max_dial_len, self.max_turn_len, self.pad_token)
        return {
            'user_dials': user_padded_dial,
            'system_dials': system_padded_dial,
            'user_turn_lens': user_turn_lens,
            'system_turn_lens': system_turn_lens,
            'dial_len': np.array(len(sample.turns))
        }


class ToTensor(object):

    def __call__(self, sample):
        # return {
        #     'user_dials': torch.from_numpy(sample['user_dials']),
        #     'system_dials': torch.from_numpy(sample['system_dials']),
        #     'user_turn_lens': torch.from_numpy(sample['user_turn_lens']),
        #     'system_turn_lens': torch.from_numpy(sample['system_turn_lens']),
        #     'dial_len': torch.from_numpy(sample['dial_len'])
        # }
        return (torch.from_numpy(sample['user_dials']),
                torch.from_numpy(sample['system_dials']),
                torch.from_numpy(sample['user_turn_lens']),
                torch.from_numpy(sample['system_turn_lens']),
                torch.from_numpy(sample['dial_len']))
