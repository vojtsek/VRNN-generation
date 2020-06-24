import argparse
import json
import os

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    valid_size = args.valid_size
    test_size = args.test_size
    train_size = 1.0 - valid_size - test_size
    assert train_size > 0.0, 'DEV + TEST must be together less than 100%'

    with open(args.data_fn, 'rt') as fd:
        data = json.load(fd)
    data = np.array(data)[np.random.permutation(len(data))]
    test_size = int(np.floor(test_size * len(data)))
    train_size = int(np.floor(train_size * len(data)))

    with open(os.path.join(args.target_dir, 'train.json'), 'wt') as train_fd, \
            open(os.path.join(args.target_dir, 'valid.json'), 'wt') as valid_fd, \
            open(os.path.join(args.target_dir, 'test.json'), 'wt') as test_fd:
        json.dump(data[:train_size].tolist(), train_fd)
        json.dump(data[train_size:-test_size].tolist(), valid_fd)
        json.dump(data[-test_size:].tolist(), test_fd)
