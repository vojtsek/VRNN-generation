from abc import ABC
from collections import Counter

class TurnRecord:
    def __init__(self,
                 turn_number,
                 turn_type,
                 prior_z_vec,
                 posterior_z_vec,
                 hyp_utterance,
                 gt_utterance):
        self.turn_number = turn_number
        self.turn_type = turn_type
        self.prior_z_vector = prior_z_vec
        self.posterior_z_vector = posterior_z_vec
        self.hyp_utterance = hyp_utterance
        self.gt_utterance = gt_utterance

    def __str__(self):
        return f'Turn {self.turn_number}, prior {self.prior_z_vector}, posterior {self.posterior_z_vector}'

    @staticmethod
    def parse(fn, records, slot_map, role):
        def strip_utterance_special_tokens(utt):
            bos_tk = '<BOS>'
            eos_tk = '<EOS>'
            return utt[utt.find(bos_tk)+len(bos_tk):utt.find(eos_tk)].strip()

        with open(fn, 'rt') as in_fd:
            current_turn_number = None
            current_turn_type = []
            prior_z_vector = None
            posterior_z_vector = None
            relative_line = 1
            current_nlu_line = ''
            for line in in_fd:
                if '--' in line:
                    records.append(TurnRecord(current_turn_number,
                                              '-'.join(current_turn_type),
                                              prior_z_vector,
                                              posterior_z_vector,
                                              hyp_utterance,
                                              gt_utterance))
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
                        if 'address' in 'line' or 'phone' in line or 'number' in line:
                            current_turn_type.append('PHONE')
                        if 'closest' in line or 'miles away' in line:
                            current_turn_type.append('WHERE')
                        if 'what city' in line:
                            current_turn_type.append('ASK-CITY')
                        if '<name> is a' in line or \
                                '<name> is located' in line:
                            current_turn_type.append('OFFER_REST')
                        if 'thank you' in line or 'bye' in line or 'welcome' in line:
                            current_turn_type.append('GOODBYE')
                        if 'there are no' in line:
                            current_turn_type.append('NO_MATCH')
                        if len(current_turn_type) == 0:
                            current_turn_type.append('OTHER')
                    if 'SYS HYP' in line:
                        hyp_utterance = strip_utterance_special_tokens(':'.join(line.split(':')[1:]))
                    if 'SYS GT' in line:
                        gt_utterance = strip_utterance_special_tokens(':'.join(line.split(':')[1:]))
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
                    if 'USER HYP' in line:
                        hyp_utterance = strip_utterance_special_tokens(':'.join(line.split(':')[1:]))
                    if 'USER GT' in line:
                        gt_utterance = strip_utterance_special_tokens(':'.join(line.split(':')[1:]))
                relative_line += 1


class Evaluator(ABC):
    def eval_from_dir(self, directory, role=None):
        raise NotImplementedError
