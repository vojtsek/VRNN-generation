import argparse
import os

from . import BleuEvaluator, ZSemanticEvaluator


def main(args):
    if not os.path.exists(args.work_dir):
        print(f'Working directory {args.work_dir} does not exist, exiting.')
        return
    evaluators = []
    metrics = args.metrics.lower()
    if 'bleu' in metrics:
        evaluators.append(BleuEvaluator())
    if 'z_semantics' in metrics:
        evaluators.append(ZSemanticEvaluator())

    for evaluator in evaluators:
        evaluator.eval_from_dir(args.work_dir, 'system')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='bleu')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--work_dir', required=True, type=str)
    args = parser.parse_args()
    main(args)
