import os
import bleu

from .commons import Evaluator, TurnRecord


class EntityEvaluator(Evaluator):
    def __init__(self, fn, database_list):
        self.bleu_score = 0
        self.fn = fn
        self.db = [entry["name"] for entry in database_list if 'name' in entry]
        self.records = []

    def eval_from_dir(self, directory, role=None):
        fn = os.path.join(directory, self.fn)
        slot_map = dict()
        TurnRecord.parse(fn, self.records, slot_map, role)

        correct = []
        for i, record in enumerate(self.records):
            for entry in self.db:
                if entry in record.gt_utterance.lower():
                    correct.append(entry in record.hyp_utterance.lower())
        acc = sum(correct)/len(correct) if len(correct) > 0 else 0
        print(acc)
        return acc
