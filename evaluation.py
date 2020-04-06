import argparse
import os
from abc import ABC

import bleu


class Evaluator(ABC):
    def eval_from_dir(self, directory, role=None):
        raise NotImplementedError


class BleuEvaluator(Evaluator):
    def __init__(self):
        self.bleu_score = 0

    def eval_from_dir(self, directory, role=None):
        with open(os.path.join(directory, f'{role}_out.txt'), 'rt') as hyp_fd,\
                open(os.path.join(directory, f'{role}_ground_truth.txt'), 'rt') as ref_fd:
            hyp = [line for line in hyp_fd]
            ref = [line for line in ref_fd]
        self.bleu_score = bleu.list_bleu([ref], hyp)
        print(self.bleu_score)


def main(args):
    if not os.path.exists(args.work_dir):
        print(f'Working directory {args.work_dir} does not exist, exiting.')
        return
    evaluators = []
    metrics = args.metrics.lower()
    if 'bleu' in metrics:
        evaluators.append(BleuEvaluator())

    for evaluator in evaluators:
        evaluator.eval_from_dir(args.work_dir, 'system')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='bleu')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--work_dir', required=True, type=str)
    args = parser.parse_args()
    main(args)
