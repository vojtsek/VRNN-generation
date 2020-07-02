import argparse
import os

from . import BleuEvaluator, ZSemanticEvaluator, ZInfoEvaluator, PPLEvaluator


def main(args):
    if not os.path.exists(args.work_dir):
        print(f'Working directory {args.work_dir} does not exist, exiting.')
        return
    evaluators = []
    metrics = args.metrics.lower()
    fn = args.fn if args.fn is not None else 'output_all.txt'
    if 'bleu' in metrics:
        evaluators.append(BleuEvaluator())
    if 'z_semantics' in metrics:
        evaluators.append(ZSemanticEvaluator(fn))
    if 'z_info' in metrics:
        evaluators.append(ZInfoEvaluator(fn))
    if 'ppl' in metrics:
        evaluators.append(PPLEvaluator(fn))

    for evaluator in evaluators:
        evaluator.eval_from_dir(args.work_dir, 'system')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='bleu')
    parser.add_argument('--work_dir', required=True, type=str)
    parser.add_argument('--fn', required=False, type=str, default=None)
    args = parser.parse_args()
    main(args)
