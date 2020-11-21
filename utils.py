import re

import numpy as np
import torch
import torch.nn.functional as torch_fun


torch.manual_seed(0)
np.random.seed(0)


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


def embed_oh(vec, size, device):
    vec = vec.unsqueeze(-1).repeat(1, 1, size[-1])
    src = torch.ones(*size).to(device)
    oh = torch.zeros(*size).to(device)

    oh.scatter_(2, vec, src)
    return oh

def zero_hidden(sizes):
    return torch.randn(*sizes)


def sample_gumbel(shape, device, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(*shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, hard=False, device=torch.device('cpu')):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    y = torch_fun.softmax(y / temperature, dim=-1)
    shape = y.shape

    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1]).to(device)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    if hard:
        y = (y_hard - y).detach() + y
    return y


def normal_sample(mu, logvar, device):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return eps * std + mu


def exponential_delta(initial, final, total_steps, current_step):
    if initial != 0:
        step = np.exp(np.log(final / initial) / total_steps)
    else:
        step = 0
    current_step = min(current_step, total_steps)
    value = initial * step ** min(current_step, total_steps)
    return value


def get_activation(activation):
    if activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'relu':
        return torch.relu
    else: # default
        return torch.tanh


def compute_ppl(prediction_scores, ground_truth, vocab, normalize_scores=None):
    total, xent = 0, 0.0
    for n, scores in enumerate(prediction_scores):
        scores = scores.squeeze(1)
        if normalize_scores == 'softmax':
            scores = torch.softmax(scores, dim=1)
        scores = scores.cpu().detach().numpy()
        total += len(scores)
        xent += np.sum([np.log(predicted_score[vocab[predicted_word]])
                        for predicted_score, predicted_word in zip(scores, ground_truth[n])])
    return xent, total

def quantize(vec, width):
    grid = np.array([width * i for i in range(int(1/ width))])
    def _round(n):
        dist = np.abs(grid-np.array([n]*len(grid)))
        return grid[np.argmin(dist)]

    vec = [_round(n) for n in vec]
    return vec
