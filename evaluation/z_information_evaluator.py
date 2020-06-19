import os
import numpy as np

from .commons import Evaluator, TurnRecord, CorpusVocab
from ..utils import tokenize


class ZInfoEvaluator(Evaluator):
    def __init__(self, fn):
        self.bleu_score = 0
        self.fn = fn
        self.data_vocab = CorpusVocab()
        self.prior_z_vocab = CorpusVocab()
        self.posterior_z_vocab = CorpusVocab()
        self.joint_vocab = CorpusVocab()
        self.records = []

    def eval_from_dir(self, directory, role='system'):
        fn = os.path.join(directory, self.fn)
        slot_map = dict()
        TurnRecord.parse(fn, self.records, slot_map, role)
        self._fill_vocabs()
        mutual_information = self.compute_mi()
        kl_qz = self.compute_kl()
        print(f'MI: {mutual_information}\nKL: {kl_qz}')

    @staticmethod
    def _make_pair(tk, z_str):
        return f'{z_str}-{tk}'

    @staticmethod
    def _tk_generator(records):
        for r in records:
            prior_z_str = ' '.join([str(tk) for tk in list(zip(*r.prior_z_vector))[1]])
            posterior_z_str = ' '.join([str(tk) for tk in list(zip(*r.posterior_z_vector))[1]])
            utt_tk = tokenize(r.gt_utterance)
            utt = [tk.strip(' ?!,.') for tk in utt_tk]
            utt = [tk for tk in utt if len(tk) > 0]
            yield None, prior_z_str, posterior_z_str
            for tk in utt:
                yield tk, prior_z_str, posterior_z_str

    def _fill_vocabs(self):
        for tk, prior, posterior in ZInfoEvaluator._tk_generator(self.records):
            if tk is None:
                self.prior_z_vocab.add_element(prior)
                self.posterior_z_vocab.add_element(posterior)
                continue
            self.data_vocab.add_element(tk)
            self.joint_vocab.add_element(ZInfoEvaluator._make_pair(tk, prior))

    def compute_mi(self):
        mi = 0
        for tk, z_prior, z_posterior in ZInfoEvaluator._tk_generator(self.records):
            joint = self.joint_vocab.element_prob(ZInfoEvaluator._make_pair(tk, z_prior))
            el_mi = np.log(joint)\
                 - np.log(self.data_vocab.element_prob(tk))\
                 - np.log(self.prior_z_vocab.element_prob(z_prior))
            mi += joint * el_mi
        return mi

    def compute_kl(self):
        kl = 0
        for tk, z_prior, z_posterior in ZInfoEvaluator._tk_generator(self.records):
            el_kl = np.log(self.posterior_z_vocab.element_prob((z_posterior)))\
                    - np.log(self.prior_z_vocab.element_prob(z_prior))
            kl += self.posterior_z_vocab.element_prob(z_posterior) * el_kl
        return kl
