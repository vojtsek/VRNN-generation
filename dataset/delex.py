import json
import os


class Delexicalizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        otgy_file = os.path.join(data_dir, 'otgy.json')
        db_file = os.path.join(data_dir, 'db.json')
        self.otgy = None
        self.db = None
        self.all_tags = set()
        if os.path.exists(otgy_file):
            with open(otgy_file, 'rt') as fd:
                self.otgy = json.load(fd)
        if os.path.exists(db_file):
            with open(db_file, 'rt') as fd:
                self.db = json.load(fd)

    def delex_utterance(self, utt):
        utt = utt.lower()
        # if self.otgy is not None:
        #     utt = self._replace_otgy(utt)
        if self.db is not None:
            utt = self._replace_db(utt)
        return utt

    def _replace_otgy(self, utt):
        for slot, values in self.otgy.items():
            for val in values:
                val = val.lower()
                if val in utt:
                    utt = utt.replace(val, self._make_tag(slot))
        return utt

    def _replace_db(self, utt):
        for entity in self.db:
            for attribute, val in entity.items():
                val = val.lower()
                if val in utt:
                    utt = utt.replace(val, self._make_tag(attribute))
        return utt

    def _make_tag(self, s):
        tag = f'<{s.lower()}>'
        self.all_tags.add(tag)
        return tag
