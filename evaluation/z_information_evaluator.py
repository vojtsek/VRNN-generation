from .commons import Evaluator, TurnRecord

class ZInfoEvaluator(Evaluator):
    def __init__(self, fn):
        self.bleu_score = 0
        self.fn = fn

    def eval_from_dir(self, directory, role=None):
        pass