import os
import numpy as np

from .commons import Evaluator, TurnRecord, CorpusVocab

class ZInfoEvaluator(Evaluator):
    def __init__(self, fn):
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
        z_lst = z_str.split()
        return [f'{i}-{zi}-{tk}' for i, zi in enumerate(z_lst)]

    @staticmethod
    def _make_vec(z_str):
        return [f'{i}-{zi}' for i, zi in enumerate(z_str.split())]

    def _fill_vocabs(self):
        for tk, prior, posterior in TurnRecord._tk_generator(self.records):
            if tk is None:
                self.prior_z_vocab.add_element(ZInfoEvaluator._make_vec(prior))
                self.posterior_z_vocab.add_element(ZInfoEvaluator._make_vec(posterior))
                continue
            self.data_vocab.add_element(tk)
            self.joint_vocab.add_element(ZInfoEvaluator._make_pair(tk, prior))

    def compute_mi(self):
        mi = 0
        for tk, z_prior, z_posterior in TurnRecord._tk_generator(self.records):
            if tk is None:
                continue
            joint = self.joint_vocab.element_prob(ZInfoEvaluator._make_pair(tk, z_prior))
#            print('joint', ZInfoEvaluator._make_pair(tk, z_prior), joint)
#            print(tk, self.data_vocab.element_prob(tk))
#            print('prior', self.prior_z_vocab.element_prob(ZInfoEvaluator._make_vec(z_prior)))
            el_mi = np.log(joint)\
                 - np.log(self.data_vocab.element_prob(tk))\
                 - np.log(self.prior_z_vocab.element_prob(ZInfoEvaluator._make_vec(z_prior)))
            mi += joint * el_mi
        return mi

    def compute_kl(self):
        kl = 0
        for tk, z_prior, z_posterior in TurnRecord._tk_generator(self.records):
            if tk is None:
                continue
            el_kl = np.log(self.posterior_z_vocab.element_prob(ZInfoEvaluator._make_vec(z_posterior)))\
                    - np.log(self.prior_z_vocab.element_prob(ZInfoEvaluator._make_vec(z_prior)))
            kl += self.posterior_z_vocab.element_prob(ZInfoEvaluator._make_vec(z_posterior)) * el_kl
        return kl
