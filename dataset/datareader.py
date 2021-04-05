import pickle
import re
from collections import Counter
import json
import os

import numpy

from ..utils import tokenize


class JSONDb:
    def __init__(self, data):
        if data is not None and isinstance(data, str) and os.path.exists(data):
            with open(data, 'rt') as fd:
                self.data = json.load(fd)
        else:
            self.data = data

    def search(self, **query):
        if 'slot' in query:
            del query['slot']

        results = []
        if len(query) == 0:
            return []
        for entry in self.data:
            match = all([col in entry and entry[col] == val for col, val in query.items()
                         if col not in ['slot'] and val != 'dontcare'])
            if match:
                results.append(entry)
        return results


class DataReader:

    def __init__(self, data=None, reader=None, saved_dialogues=None,
                 delexicalizer=None, train=.8, valid=.1, db_file=None):
        self._dialogues = []
        self.max_dial_len = 0
        self.max_turn_len = 0
        self.max_slu_len = 0
        self.all_actions = set()
        if db_file is not None and os.path.exists(db_file):
            self.db = JSONDb(db_file)
        else:
            self.db = None
        self.delexicalizer = delexicalizer
        self.all_words = Counter()
        if saved_dialogues is not None:
            with open(saved_dialogues, 'rb') as fd:
                print('Loading data from "{}"'.format(saved_dialogues))
                self._dialogues = pickle.load(fd)
                self.length = len(self._dialogues)
        else:
            self.reader = reader
            self._parse_data(data)
        self.train = train
        self.valid = valid
        self.permutation = list(range(len(self._dialogues)))

    def permute(self, seed=0):
        numpy.random.seed(seed)
        self.permutation = numpy.random.permutation(len(self.dialogues))

    @property
    def dialogues(self):
        dials = []
        for idx in self.permutation:
            dials.append(self._dialogues[idx])
        return dials

    def _parse_data(self, data):
        db = self.db
        for d in self.reader.parse_dialogues(data, self.delexicalizer):
            if not len(d.turns) > 0:
                continue
            if any([t.user is None or t.system is None for t in d.turns]):
                continue

            self._dialogues.append(d)
            state = {}
            if hasattr(self.reader, 'db_data'):
                db = JSONDb(self.reader.db_data)
            for t in d.turns:
                self.all_words.update(t.user)
                self.all_words.update(t.system)
                self.all_words.update([s.val for s in t.usr_slu])
                self.all_actions.update(set(t.system_nlu))

                if db is not None:
                    for s in t.usr_slu:
                        state[s.name] = s.val
                    db_result = db.search(**state)
                    t.db_len = len(db_result)
                    t.db_result = db_result
                else:
                    t.db_len = 0

            self.max_dial_len = max(self.max_dial_len, len(d.turns))
            self.max_turn_len = max(self.max_turn_len, max([max(len(t.user), len(t.system)) for t in d.turns]))
            self.max_slu_len = max(self.max_slu_len, max([max(len(t.usr_slu), len(t.system_nlu)) for t in d.turns]))
        self.length = len(self._dialogues)

    def apply_to_dialogues(self, fun):
        for d in self.dialogues:
            fun(d)

    def apply_to_turns(self, fun):
        print('Applying {}'.format(fun))
        for dialogue in self.dialogues:
            for turn in dialogue.turns:
                fun(turn)

    def _turns(self, dials):
        for d in dials:
            for t in d.turns:
                yield t

    def turns_from_chunk(self, chunk_idxs):
        for i in chunk_idxs:
            d = self._dialogues[self.permutation[i]]
            for turn in d.turns:
                yield turn

    @property
    def turns(self):
        return self._turns(self.dialogues)
    
    @property
    def test_set(self):
        train_size = round(self.train * self.length)
        valid_size = round(self.valid * self.length)
        return self.dialogues[train_size+valid_size:]

    @property
    def train_set(self):
        train_size = round(self.train * self.length)
        return self.dialogues[:train_size]

    @property
    def valid_set(self):
        train_size = round(self.train * self.length)
        valid_size = round(self.valid * self.length)
        return self.dialogues[train_size:(train_size+valid_size)]
    
    def user_utterances(self):
        for t in self.turns:
            yield t.user
    
    def save_dialogues(self, output_fn):
        with open(output_fn, 'wb') as fd:
            pickle.dump(self.dialogues, fd)


class Dialogue:
    
    def __init__(self):
        self.turns = []

    def add_turn(self, turn):
        self.turns.append(turn)


class Turn:

    def __init__(self, delexicalizer=None):
        self.user = None
        self.system = None
        self.usr_slu = None
        self.state = None
        self.system_nlu = None
        self.parse = None
        self.intent = None
        self.delexicalizer = delexicalizer
    
    def add_user(self, utt):
        self.orig_user = tokenize(utt)
      #  if self.delexicalizer is not None:
      #      utt, _ = self.delexicalizer.delex_utterance(utt)
        utt = utt.lower()
        utt = re.sub(r'\d+', '<NUM>', utt)
        self.user = tokenize(utt)

    def add_system(self, utt):
        found_tags = []
        self.orig_system = tokenize(utt)
        if self.delexicalizer is not None:
            utt, found_tags = self.delexicalizer.delex_utterance(utt)
        self.system = tokenize(utt)
        self.system_nlu = list(set(found_tags))

    def add_usr_slu(self, usr_slu):
        self.usr_slu = self._process_slu(usr_slu)

    def add_state(self, state):
        self.state = self._process_slu(state)

    def add_sys_slu(self, sys_slu):
        self.system_nlu = sys_slu

    def add_intent(self, intent):
        self.intent = intent

    def _process_slu(self, slu):
        if self.delexicalizer is not None:
            for s in slu:
                s.val, _ = self.delexicalizer.delex_utterance(s.val)
        return slu


class Slot:

    def __init__(self, name, val, intent):
        self.name = name
        self.val = val
        self.intent = intent


class CamRestReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data, delexicalizer=None):
        for dial in data:
            dialogue = Dialogue()
            turns = dial['dial']
            last_state = {}
            for t in turns:
                turn = Turn(delexicalizer)
                turn.add_user(t['usr']['transcript'])
                turn.add_system(t['sys']['sent'])
                state = self.parse_slu(t['usr']['slu'])
                slu = []
                for s in state:
                    if s.name not in last_state or last_state[s.name] != s.val:
                        slu.append(s)
                    last_state[s.name] = s.val
                turn.add_usr_slu(slu)
                turn.add_state(state)
                intent_counter = Counter()
                for slot in slu:
                    intent_counter[slot.intent] += 1
                if len(intent_counter) > 0:
                    turn.add_intent(intent_counter.most_common(1)[0][0])
                else:
                    turn.add_intent(None)
                dialogue.add_turn(turn)
            yield dialogue

    def parse_slu(self, slu):
        usr_slu = []
        for da in slu:
            for s in da['slots']:
                slot = Slot(s[0], s[1], da['act'])
                usr_slu.append(slot)
        return usr_slu


class MultiWOZReader:
    
    def __init__(self, allowed_domains, max_allowed_len):
        self.allowed_domains = allowed_domains
        self.max_allowed_len = max_allowed_len

    def parse_dialogues(self, data, delexicalizer=None):
        for dial in data:
            dialogue = Dialogue()
            turns = dial['log']
            i = 0
            max_turn_len = 0
            for t in turns:
                i += 1

                text = t['text'].strip().replace('\n', ' ')
                if len(text) < 1:
                    continue
                if i % 2 == 1:
                    turn = Turn(delexicalizer)
                    if 'dialog_act' in t:
                        slu = self.parse_slu(t['dialog_act'])
                    else:
                        slu = []

                    turn.add_usr_slu(slu)
                    turn.add_user(text)
                    max_turn_len = max(max_turn_len, len(text.split()))
                else:
                    turn.add_system(text)
                    turn.add_usr_slu(self.parse_meta(t['metadata']))
                    if 'sys_action' in t:
                        turn.add_sys_slu(t['sys_action'].split(','))
                    else:
                        turn.add_sys_slu([])
                    max_turn_len = max(max_turn_len, len(text.split()))
                    dialogue.add_turn(turn)
                    continue

                # if not 'dialog_act' in t:
                #     continue
                # slu = self.parse_slu(t['dialog_act'])
                # if len(slu) == 0:
                #     continue
                # turn.add_usr_slu(slu)
                # intent_counter = Counter()
                # for slot in slu:
                #     intent_counter[slot.intent] += 1
                # if len(intent_counter) > 0:
                #     turn.add_intent(intent_counter.most_common(1)[0][0])
                # else:
                #     turn.add_intent(None)

            if max_turn_len < self.max_allowed_len:
                yield dialogue

    def parse_meta(self, meta):
        parsed = []
        for domain, state in meta.items():
            for slot, val in state['semi'].items():
                if len(val) > 0 and val not in ['not mentioned', 'dontcare']:
                    parsed.append(Slot(slot.lower(), val, 'inform'))
        return parsed

    def parse_slu(self, slu):
        usr_slu = []
        print(slu)
        for intent_domain, val in slu.items():
            domain, intent = intent_domain.split('-')
            intent = intent.lower()
            domain = domain.lower()
            if domain not in self.allowed_domains:
                continue
            for s in val:
                slot = Slot(s[0].lower(), s[1], intent)
            usr_slu.append(slot)
        return usr_slu


class MovieReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data):
        for dial in data['SearchScreeningEvent']:
            dialogue = Dialogue()
            text, slu = self.extract_turn(dial['data'])
            text = text.strip().replace('\n', '')
            print(text, slu)
            turn = Turn()
            turn.add_user(text)
            turn.add_system('dummy')
            turn.add_usr_slu(slu)
            dialogue.add_turn(turn)
            yield dialogue

    def extract_turn(self, data):
        text = ''.join([tk['text'] for tk in data])
        entities = [tk for tk in data if 'entity' in tk]
        slu = [ Slot(e['entity'].lower(), e['text'], 'unk') for e in entities]
        return text, slu


class AtisReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data):
        for dial in data['rasa_nlu_data']['common_examples']:
            dialogue = Dialogue()
            text = dial['text'].strip().replace('\n', '')
            turn = Turn()
            turn.add_user(text)
            turn.add_system('dummy')
            print(text, dial['intent'])
            turn.add_usr_slu(Slot(None, None, dial['intent']))
            dialogue.add_turn(turn)
            yield dialogue

    def extract_turn(self, data):
        text = ''.join([tk['text'] for tk in data])
        entities = [tk for tk in data if 'entity' in tk]
        slu = [ Slot(e['entity'].lower(), e['text'], 'unk') for e in entities]
        return text, slu


class CarsluReader:
    
    def __init__(self):
        pass

    def parse_dialogues(self, data):
        for t in data:
            dialogue = Dialogue()
            turn = Turn()
            if 'text' not in t:
                continue
            turn.add_user(t['text'])
            turn.add_system('dummy')
            intent, slots = t['slu']
            print(slots)
            slu = []
            for sl in slots:
                if len(sl) == 1:
                    slu.append(Slot(sl[0], sl[0], intent))
                else:
                    slu.append(Slot(sl[0], sl[1], intent))
            turn.add_usr_slu(slu)
            dialogue.add_turn(turn)
            yield dialogue


class SMDReader:

    def parse_dialogues(self, data, delexicalizer=None):
        self.db_data = []
        for dial in data:
            dialogue = Dialogue()
            turns = dial['dialogue']
            if dial['scenario']['kb']['items'] is not None:
                self.db_data.extend(dial['scenario']['kb']['items'])
            last_state = {}
            for t in turns:
                tt = t['turn']
                if tt == 'driver':
                    turn = Turn(delexicalizer)
                    turn.add_user(t['data']['utterance'])
                elif turn is not None:
                    turn.add_system(t['data']['utterance'])
                    slu = self.parse_slu(t['data']['slots'])
                    turn.add_usr_slu(slu)
                    dialogue.add_turn(turn)
                    turn = None

                # state = self.parse_slu(t['usr']['slu'])
                # slu = []
                # for s in state:
                #     if s.name not in last_state or last_state[s.name] != s.val:
                #         slu.append(s)
                #     last_state[s.name] = s.val
                # turn.add_state(state)
            yield dialogue

    def parse_slu(self, slots):
        slu = [Slot(k, v, 'dummy') for k, v in slots.items()]
        return slu


class DDReader:

    def parse_dialogues(self, data, delexicalizer=None):
        for dial in data:
            dialogue = Dialogue()
            for t in dial:
                turn = Turn(delexicalizer)
                turn.add_user(t['usr'])
                turn.add_system(t['system'])
                turn.add_usr_slu([])
                turn.add_sys_slu([])
                dialogue.add_turn(turn)

            yield dialogue

