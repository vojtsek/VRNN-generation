from abc import ABC
from collections import Counter

from ..utils import tokenize


def parse_query(query, onto):
    def _delex(tk):
        for slot, vals in onto.items():
            if tk in vals:
                return slot
        return tk
    query = [tk for tk in query.split() if tk not in ['query', 'dontcare', 'address', 'phone', 'number', 'postcode']]
    if onto is not None and len(onto) > 0:
        query = [_delex(tk) for tk in query]
    return query


class TurnRecord:
    def __init__(self,
                 turn_number,
                 turn_type,
                 prior_z_vec,
                 posterior_z_vec,
                 hyp_utterance,
                 gt_utterance,
                 sys_nlu,
                 gt_query=None,
                 hyp_query=None):
        self.turn_number = turn_number
        self.turn_type = turn_type
        self.prior_z_vector = prior_z_vec
        self.posterior_z_vector = posterior_z_vec
        self.hyp_utterance = hyp_utterance
        self.gt_utterance = gt_utterance
        self.sys_nlu = sys_nlu
        sys_nlu = [action for action in sys_nlu if 'booking' not in action.lower()]
        self.gt_query = gt_query
        self.hyp_query = hyp_query
        # self.turn_type = sys_nlu[0] if len(sys_nlu) > 0 else 'unk'

    def __str__(self):
        return f'Turn {self.turn_number}, prior {self.prior_z_vector}, posterior {self.posterior_z_vector}'

    @staticmethod
    def parse(fn, records, slot_map, role, onto=None):
        def strip_utterance_special_tokens(utt):
            bos_tk = '<BOS>'
            eos_tk = '<EOS>'
            return utt[utt.find(bos_tk)+len(bos_tk):utt.find(eos_tk)].strip()

        if onto is None:
            onto = dict()

        with open(fn, 'rt') as in_fd:
            current_turn_number = None
            current_turn_type = []
            prior_z_vector = None
            gt_query = None
            hyp_query = None
            posterior_z_vector = None
            relative_line = 1
            current_nlu_line = ''
            for line in in_fd:
                if '---' in line:
                    records.append(TurnRecord(current_turn_number,
                                              '-'.join(current_turn_type),
                                              prior_z_vector,
                                              posterior_z_vector,
                                              hyp_utterance,
                                              gt_utterance,
                                              sys_nlu,
                                              gt_query,
                                              hyp_query))
                    current_turn_number = None
                    current_turn_type = []
                    prior_z_vector = None
                    posterior_z_vector = None
                    gt_query = None
                    hyp_query = None
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
                    if 'SYS HYP' in line:
                        hyp_utterance = strip_utterance_special_tokens(':'.join(line.split(':')[1:]))
                        if 'query' in line:
                            hyp_query = parse_query(strip_utterance_special_tokens(':'.join(line.split(':')[1:])), onto)
                    if 'SYS GT' in line:
                        if 'query' in line:
                            gt_query = parse_query(strip_utterance_special_tokens(':'.join(line.split(':')[1:])), onto)
                            current_turn_type.append('QUERY')
                        elif 'address' in 'line' or 'phone' in line or 'number' in line:
                            current_turn_type.append('PHONE')
                        elif 'closest' in line or 'miles away' in line:
                            current_turn_type.append('WHERE')
                        elif 'what city' in line:
                            current_turn_type.append('ASK-CITY')
                        elif '<name> is a' in line or \
                                '<name> is located' in line:
                            current_turn_type.append('OFFER_REST')
                        elif 'thank you' in line or 'bye' in line or 'welcome' in line:
                            current_turn_type.append('GOODBYE')
                        elif 'there are no' in line:
                            current_turn_type.append('NO_MATCH')
                        elif len(current_turn_type) == 0:
                            current_turn_type.append('OTHER')

                        gt_utterance = strip_utterance_special_tokens(':'.join(line.split(':')[1:]))
                    if 'SYS NLU' in line:
                        sys_nlu = strip_utterance_special_tokens(':'.join(line.split(':')[1:])).split()
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

    @staticmethod
    def _tk_generator(records):
        for r in records:
            if r.prior_z_vector is None:
                continue
            prior_z_str = ' '.join([str(tk) for tk in list(zip(*r.prior_z_vector))[1]])
            posterior_z_str = ' '.join([str(tk) for tk in list(zip(*r.posterior_z_vector))[1]])
            utt_tk = tokenize(r.gt_utterance)
            utt = [tk.strip(' ?!,.') for tk in utt_tk]
            utt = [tk for tk in utt if len(tk) > 0]
            yield None, prior_z_str, posterior_z_str
            for tk in utt:
                yield tk, prior_z_str, posterior_z_str

class Evaluator(ABC):
    def eval_from_dir(self, directory, role=None):
        raise NotImplementedError


class CorpusVocab:
    def __init__(self):
        self.vocab = Counter()

    def add_element(self, elem):
        if not isinstance(elem, list):
            elem = [elem]
        self.vocab.update(elem)

    def element_prob(self, elem):
        if elem not in self.vocab:
            prob = 1e-15
        else:
            prob = self.vocab[elem] / sum(self.vocab.values())
        return prob
