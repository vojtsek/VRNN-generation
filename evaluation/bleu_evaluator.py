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
        for i, r in enumerate(self.records):
            if (i < len(self.records) - 1):
                hyp.append(self.records[i].hyp_utterance)
                ref.append(self.records[i+1].gt_utterance)
        self.bleu_score = bleu.list_bleu([ref], hyp)
        print('BLEU', self.bleu_score)
        return self.bleu_score
