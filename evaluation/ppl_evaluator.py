import os
import pickle

import numpy as np

from .commons import Evaluator, TurnRecord
from ..utils import tokenize


class PPLEvaluator(Evaluator):
    def __init__(self, fn):
        self.fn = fn
        self.vocab = None
        self.scores = None
        self.records = []

    def eval_from_dir(self, directory, role='system'):
        fn = os.path.join(directory, self.fn)
        slot_map = dict()
        TurnRecord.parse(fn, self.records, slot_map, role)
        with open(os.path.join(directory, 'w2id_vocab.pkl'), 'rb') as vocab_fd, \
                open(os.path.join(directory, 'raw_scores.pkl'), 'rb') as scores_fd:
            self.vocab = pickle.load(vocab_fd)
            self.scores = pickle.load(scores_fd)
        ppl = self._compute_ppl()
        print(f'Perplexity: {ppl}')

    def _compute_ppl(self):
        d = -1
        xent_total, count = 0, 0
        for t, record in enumerate(self.records):
            turn_no = record.turn_number
            d += turn_no == 1
            xent, n = self.xent_single_turn(record.gt_utterance, self.scores[d][turn_no - 1])
            xent_total += xent
            count += n

        return np.exp(xent_total / count)

    def xent_single_turn(self, gold, scores):
        log_probs = [scores[i][0][self.vocab[tk.strip(' ?!,.')]] for
                           i, tk in enumerate(tokenize(gold))]

        return - sum(log_probs), len(log_probs)
