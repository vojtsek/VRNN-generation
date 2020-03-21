import numpy as np


def tokenize(utt):
    return utt.split()


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
