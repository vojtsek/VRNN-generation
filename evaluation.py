import argparse
import os
import copy
from itertools import groupby
from collections import Counter
from abc import ABC

import bleu


class Evaluator(ABC):
    def eval_from_dir(self, directory, role=None):
        raise NotImplementedError


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


class TurnRecord:
    def __init__(self, turn_number, turn_type, prior_z_vec, posterior_z_vec):
        self.turn_number = turn_number
        self.turn_type = turn_type
        self.prior_z_vector = prior_z_vec
        self.posterior_z_vector = posterior_z_vec

    def __str__(self):
        return f'Turn {self.turn_number}, prior {self.prior_z_vector}, posterior {self.posterior_z_vector}'


class ZSemanticEvaluator(Evaluator):

    def __init__(self):
        self.bleu_score = 0

    def eval_from_dir(self, directory, role=None):
        with open(os.path.join(directory, f'output_all.txt'), 'rt') as in_fd:
            current_turn_number = None
            current_turn_type = []
            slot_map = dict()
            prior_z_vector = None
            posterior_z_vector = None
            relative_line = 1
            records = []
            current_nlu_line = ''
            for line in in_fd:
                if '--' in line:
                    records.append(TurnRecord(current_turn_number,
                                              '-'.join(current_turn_type),
                                              prior_z_vector,
                                              posterior_z_vector))
                    current_turn_number = None
                    current_turn_type = []
                    prior_z_vector = None
                    posterior_z_vector = None
                if 'Turn' in line:
                    line = line.split()
                    current_turn_number = int(line[1])
                if 'prior Z:' in line:
                    line = line.split()
                    prior_z_vector = [(i, int(n)) for i, n in enumerate(line[2:])]
                if 'post Z:' in line:
                    line = line.split()
                    posterior_z_vector = [(i, int(n)) for i, n in enumerate(line[2:])]
                # if 'user Z:' in line:
                #     line = line.split()
                #     posterior_z_vector = [int(n) for n in line[2:]]

                if role == 'system':
                    if 'SYS HYP:' in line:
                        if 'address' in line:
                            current_turn_type.append('ADDRESS')
                        if 'phone' in line or 'number' in line:
                            current_turn_type.append('PHONE')
                        if '<name> is a' in line or \
                                '<name> is located' in line:
                            current_turn_type.append('OFFER_REST')
                        if 'thank you' in line or 'bye' in line or 'welcome' in line:
                            current_turn_type.append('GOODBYE')
                        if 'there are no' in line:
                            current_turn_type.append('NO_MATCH')
                else:
                    if 'Turn' in line:
                        relative_line = 0
                    if relative_line == 2:
                        current_nlu_line = line
                    if 'user Z' in line:
                        line = line.split()
                        z_vector = [int(n) for n in line[2:]]
                        for val in current_nlu_line.split():
                            if val not in ['<BOS>', '<EOS>']:
                                if val not in slot_map:
                                    slot_map[val] = Counter()
                                slot_map[val].update([' '.join([str(z) for z in z_vector])])
                relative_line += 1

        if role == 'system':
            records = sorted(records, key=lambda r: r.turn_type)
            for t_tpe, records in groupby(records, key=lambda r: r.turn_type):
                r = copy.deepcopy(records)
                print(t_tpe, len(list(r)))
                t_counter = Counter()
                for record in records:
                    t_counter.update([str(i) for i in record.prior_z_vector])
                print(t_counter.most_common(5))
        else:
            for val, zs in slot_map.items():
                print(val, zs.most_common(5))


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
