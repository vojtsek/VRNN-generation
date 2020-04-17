import re

import numpy as np
import torch
import torch.nn.functional as torch_fun


def tokenize(utt):
    utt = utt.replace('\'s', ' is')
    utt = utt.replace('\'d', ' would')
    utt = utt.replace('\'m', ' am')
    utt = utt.replace('aren\'t', 'are not')
    utt = utt.replace('isn\'t', 'is not')
    utt = utt.replace('don\'t', 'do not')
    utt = re.sub(r'(\w)\.', r'\1 .', utt, flags=re.DOTALL)
    utt = re.sub(r'(\w)\?', r'\1 ?', utt, flags=re.DOTALL)
    utt = re.sub(r'(\w)!', r'\1 !', utt, flags=re.DOTALL)
    utt = re.sub(r'(\w),', r'\1 ,', utt, flags=re.DOTALL)


    return [tk.lower().strip('\'') for tk in utt.split()]


def pad_turn(tokens, max_turn_len, pad):
    padded = tokens
    padded.extend([pad] * (max_turn_len - len(tokens)))
    return np.array(padded)


def pad_dial(turns, max_dial_len, max_turn_len, pad):
    padded = []
    turn_lens = []
    remaining = max_dial_len
    for t in turns:
        turn_lens.append(len(t))
        padded.append(pad_turn(t, max_turn_len, pad))
        remaining -= 1
    for _ in range(remaining):
        padded.append(pad_turn([], max_turn_len, pad))
        turn_lens.append(0)
    return np.array(padded), np.array(turn_lens)


def zero_hidden(sizes):
    return torch.randn(*sizes)


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(*shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, hard=False):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape)
    y = torch_fun.softmax(y / temperature, dim=-1)
    shape = y.shape

    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    if hard:
        y = (y_hard - y).detach() + y
    return y




def normal_sample(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu
