import argparse
import json
import os

from . import BleuEvaluator, ZSemanticEvaluator, ZInfoEvaluator, PPLEvaluator, EntityEvaluator, ClfEvaluator


def main(args):
    if not os.path.exists(args.work_dir):
        print(f'Working directory {args.work_dir} does not exist, exiting.')
        return
    evaluators = []
    metrics = args.metrics.lower()
    fn = args.fn if args.fn is not None else 'output_all.txt'
    if 'bleu' in metrics:
        evaluators.append(BleuEvaluator(fn))
    if 'z_semantics' in metrics:
        evaluators.append(ZSemanticEvaluator(fn, args.test_fn))
    if 'z_info' in metrics:
        evaluators.append(ZInfoEvaluator(fn))
    if 'clf' in metrics:
        evaluators.append(ClfEvaluator(fn, args.test_fn))
    if 'ppl' in metrics:
        evaluators.append(PPLEvaluator(fn))
    if 'ent' in metrics:
        with open(args.db_file, 'rt') as inf:
            db = json.load(inf)
        evaluators.append(EntityEvaluator(fn, db))

    for evaluator in evaluators:
        evaluator.eval_from_dir(args.work_dir, args.role)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='bleu')
    parser.add_argument('--work_dir', required=True, type=str)
    parser.add_argument('--fn', required=False, type=str, default=None)
    parser.add_argument('--test_fn', required=False, type=str)
    parser.add_argument('--role', required=False, type=str, default='system')
    parser.add_argument('--db_file', required=False, type=str)
    args = parser.parse_args()
    main(args)
