import json
import os
import re


class Delexicalizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        otgy_file = os.path.join(data_dir, 'otgy.json')
        db_file = os.path.join(data_dir, 'db.json')
        self.otgy = None
        self.db = None
        self.found_tags = None
        self.all_tags = set()
        if os.path.exists(otgy_file):
            with open(otgy_file, 'rt') as fd:
                self.otgy = self._load_otgy(fd)
        if os.path.exists(db_file):
            with open(db_file, 'rt') as fd:
                self.db = json.load(fd)

    def _load_otgy(self, fd):
        otgy = json.load(fd)
        for ent in otgy:
            ent = ent.replace(' ', '-')
            otgy[ent] = [str(val) for val in otgy[ent]]
        return otgy

    def delex_utterance(self, utt):
        utt = utt.lower()
        self.found_tags = []
        if self.otgy is not None:
            utt = self._replace_otgy(utt)
        if self.db is not None:
            utt = self._replace_db(utt)
        utt = re.sub(r'\d+', '<NUM>', utt)
        return utt, self.found_tags

    def _replace_otgy(self, utt):
        for slot, values in self.otgy.items():
            for val in values:
                val = val.lower()
                if val in utt:
                    utt = re.sub(r'' + val + '\w*', self._make_tag(slot), utt)
                    break
        return utt

    def _replace_db(self, utt):
        for entity in self.db:
            for attribute, val in entity.items():
                if attribute in ['area', 'food', 'pricerange']:
                    continue
                val = val.lower()
                if val in utt:
                    self.found_tags.append(self._make_tag(attribute))
                    utt = utt.replace(val, self._make_tag(attribute))
        return utt

    def _make_tag(self, s):
        tag = f'<{s.lower()}>'
        self.all_tags.add(tag)
        return tag
