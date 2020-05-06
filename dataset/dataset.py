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

    def __init__(self, embeddings: Embeddings, db_max: int):
        self.db_max = db_max
        self.embeddings = embeddings

    def __call__(self, sample):
        for t in sample.turns:
            t.user = [self.embeddings.w2id[tk] for tk in [Embeddings.BOS] + t.user + [Embeddings.EOS]]
            t.system = [self.embeddings.w2id[tk] for tk in [Embeddings.BOS] + t.system + [Embeddings.EOS]]
            t.usr_slu = [self.embeddings.w2id[Embeddings.BOS]] +\
                        [self.embeddings.w2id[s.val] for s in t.usr_slu[:len(t.user)-2]] + [self.embeddings.w2id[Embeddings.EOS]]
            t.sys_nlu = [self.embeddings.w2id[s] for s in [Embeddings.BOS] + t.system_nlu + [Embeddings.EOS]]
            t.db = min(t.db_len, self.db_max)
        return sample


class Padding(object):

    def __init__(self, pad_token: int, max_dial_len: int, max_turn_len: int, max_slu_len: int = None):
        self.max_dial_len = max_dial_len
        self.max_turn_len = max_turn_len
        self.max_slu_len = max_slu_len if max_slu_len is not None else max_turn_len
        self.pad_token = pad_token

    def __call__(self, sample):
        user_turns = [t.user for t in sample.turns]
        system_turns = [t.system for t in sample.turns]
        usr_nlu_turns = [t.usr_slu for t in sample.turns]
        sys_nlu_turns = [t.sys_nlu for t in sample.turns]
        db_turns = [[t.db] for t in sample.turns]
        user_padded_dial, user_turn_lens = pad_dial(
            user_turns, self.max_dial_len, self.max_turn_len, self.pad_token)
        system_padded_dial, system_turn_lens = pad_dial(
            system_turns, self.max_dial_len, self.max_turn_len, self.pad_token)
        usr_nlu_padded_dial, usr_nlu_turn_lens = pad_dial(
            usr_nlu_turns, self.max_dial_len, self.max_slu_len, self.pad_token)
        sys_nlu_padded_dial, sys_nlu_turn_lens = pad_dial(
            sys_nlu_turns, self.max_dial_len, self.max_slu_len, self.pad_token)
        db_padded_dial, _ = pad_dial(
            db_turns, self.max_dial_len, 1, self.pad_token)

        return {
            'user_dials': user_padded_dial,
            'system_dials': system_padded_dial,
            'usr_nlu_dials': usr_nlu_padded_dial,
            'sys_nlu_dials': sys_nlu_padded_dial,
            'user_turn_lens': user_turn_lens,
            'system_turn_lens': system_turn_lens,
            'usr_nlu_turn_lens': usr_nlu_turn_lens,
            'sys_nlu_turn_lens': sys_nlu_turn_lens,
            'db_dials': db_padded_dial,
            'dial_len': np.array(len(sample.turns))
        }


class ToTensor(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return (torch.from_numpy(sample['user_dials']).to(self.device),
                torch.from_numpy(sample['system_dials']).to(self.device),
                torch.from_numpy(sample['usr_nlu_dials']).to(self.device),
                torch.from_numpy(sample['sys_nlu_dials']).to(self.device),
                torch.from_numpy(sample['user_turn_lens']).to(self.device),
                torch.from_numpy(sample['system_turn_lens']).to(self.device),
                torch.from_numpy(sample['usr_nlu_turn_lens']).to(self.device),
                torch.from_numpy(sample['sys_nlu_turn_lens']).to(self.device),
                torch.from_numpy(sample['db_dials']).to(self.device),
                torch.from_numpy(sample['dial_len']).to(self.device))

