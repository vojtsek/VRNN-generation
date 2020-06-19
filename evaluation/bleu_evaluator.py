import os
import bleu

from .commons import Evaluator


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
