import os
import numpy as np

from .commons import Evaluator, TurnRecord, CorpusVocab

class ZInfoEvaluator(Evaluator):
    def __init__(self, fn):
        self.fn = fn
        self.data_vocab = CorpusVocab()
        self.joint_vocabs = None
        self.records = []
        self.prior_z_vocabs = None
        self.posterior_z_vocabs = None

    def eval_from_dir(self, directory, role='system'):
        fn = os.path.join(directory, self.fn)
        slot_map = dict()
        TurnRecord.parse(fn, self.records, slot_map, role)
        self._fill_vocabs()
        mutual_information = self.compute_mi()
        #kl_qz = self.compute_kl()
        kl_qz = 0
        print(f'MI: {mutual_information}\nKL: {kl_qz}')
        return mutual_information

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
                prior_vec = ZInfoEvaluator._make_vec(prior)
                posterior_vec = ZInfoEvaluator._make_vec(posterior)
                if self.prior_z_vocabs is None:
                    self.prior_z_vocabs = [CorpusVocab() for _ in range(len(prior_vec))]
                    self.posterior_z_vocabs = [CorpusVocab() for _ in range(len(prior_vec))]
                    self.joint_vocabs = [CorpusVocab() for _ in range(len(prior_vec))]
                for n, (pr_el, po_el) in enumerate(zip(prior_vec, posterior_vec)):
                    self.prior_z_vocabs[n].add_element(pr_el)
                    self.posterior_z_vocabs[n].add_element(po_el)
                continue
            self.data_vocab.add_element(tk)
            for n, el in enumerate(ZInfoEvaluator._make_pair(tk, prior)):
                self.joint_vocabs[n].add_element(el)

    def compute_mi(self):
        mi = [0 for _ in range(len(self.prior_z_vocabs))]
#        print(self.joint_vocabs[0].vocab)
#        print(self.prior_z_vocabs[0].vocab)
        for tk, z_prior, z_posterior in TurnRecord._tk_generator(self.records):
            if tk is None:
                continue

            joint_entries = ZInfoEvaluator._make_pair(tk, z_prior)
            prior_entries = ZInfoEvaluator._make_vec(z_prior)
            for n in range(len(self.prior_z_vocabs)):
                joint = self.joint_vocabs[n].element_prob(joint_entries[n])
                el_mi = np.log(joint)\
                        - np.log(self.data_vocab.element_prob(tk))\
                        - np.log(self.prior_z_vocabs[n].element_prob(prior_entries[n]))
                mi[n] += joint * el_mi
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
