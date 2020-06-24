import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--otgy', type=str)
    # MultiWOZ hotel-book stay,hotel-book people,hotel-stars,train-book people
    # SMD distance,temperature
    parser.add_argument('--remove_fields',
                        type=str,
                        default='hotel-book stay,hotel-book people,hotel-stars,train-book people,distance,temperature')
    args = parser.parse_args()
    with open(args.otgy, 'rt') as fd:
        data = json.load(fd)
    for field in args.remove_fields.split(','):
        if field in data:
            del data[field]
    with open(args.otgy, 'wt') as fd:
        json.dump(data, fd)
