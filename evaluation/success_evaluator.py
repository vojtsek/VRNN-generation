import os
import bleu
import json

from .commons import Evaluator, TurnRecord


class SuccessEvaluator(Evaluator):
    def __init__(self, fn, onto):
        self.bleu_score = 0
        self.fn = fn
        self.onto = onto
        if onto is not None:
            with open(onto, 'rt') as f:
                self.onto = json.load(f)
        self.records = []

    def eval_from_dir(self, directory, role=None):
        fn = os.path.join(directory, self.fn)
        slot_map = dict()
        TurnRecord.parse(fn, self.records, slot_map, role, self.onto)

        fn = tp = fp = 0
        correct = total = 0
        for record in self.records:
            if record.hyp_query is None and record.gt_query is not None:
                fn += 1
            if record.hyp_query is not None and record.gt_query is None:
                fp += 1
            if record.hyp_query is not None and record.gt_query is not None:
                total += 1
                tp += 1
                hyp = sorted(record.hyp_query)
                gt = sorted(record.gt_query)
                correct += int(all([t1 == t2 for t1, t2 in zip(hyp, gt)]))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        acc = correct / total
        print(f'P: {precision}\tR: {recall}\tF1: {f1}')
        print(f'Acc: {acc}')

