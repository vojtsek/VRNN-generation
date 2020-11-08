import os
import bleu

from .commons import Evaluator, TurnRecord


class BleuEvaluator(Evaluator):
    def __init__(self, fn):
        self.bleu_score = 0
        self.fn = fn
        self.records = []

    def eval_from_dir(self, directory, role=None):
        fn = os.path.join(directory, self.fn)
        slot_map = dict()
        TurnRecord.parse(fn, self.records, slot_map, role)

        hyp = []
        ref = []
        for record in self.records:
            hyp.append(record.hyp_utterance)
            ref.append(record.gt_utterance)
        self.bleu_score = bleu.list_bleu([ref], hyp)
        print('BLEU', self.bleu_score)
        return self.bleu_score
