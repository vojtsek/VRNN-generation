import os
import numpy as np
from collections import Counter

from .commons import Evaluator, TurnRecord, CorpusVocab


class ZInfoEvaluator(Evaluator):
    def __init__(self, fn):
        self.fn = fn
        self.data_vocab = CorpusVocab()
        self.joint_vocabs = None
        self.records = []
        self.prior_z_vocabs = None
        self.posterior_z_vocabs = None
        self.z_vecs = []

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
                    self.z_vecs = [[] for _ in range(len(prior_vec))]
                for n, (pr_el, po_el) in enumerate(zip(prior_vec, posterior_vec)):
                    self.prior_z_vocabs[n].add_element(pr_el)
                    self.posterior_z_vocabs[n].add_element(po_el)
                    self.z_vecs[n].append(pr_el)
                continue
            self.data_vocab.add_element(tk)
            for n, el in enumerate(ZInfoEvaluator._make_pair(tk, prior)):
                self.joint_vocabs[n].add_element(el)

        print('Z1 vs Z2', ZInfoEvaluator.compute_mi_two_vec(self.z_vecs[0], self.z_vecs[1]))

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

    @staticmethod
    def compute_mi_two_vec(vec1, vec2):
        joint = Counter()
        pr1 = Counter()
        pr2 = Counter()

        for el1, el2 in zip(vec1, vec2):
            joint.update([(el1, el2)])
            pr1.update([el1])
            pr2.update([el2])
        mi = 0
        for el1, el2 in zip(vec1, vec2):
            j_pr = joint[(el1, el2)] / sum(j_pr.values())
            p1 = pr1[el1] / sum(pr1.values())
            p2 = pr2[el2] / sum(pr2.values())
            mi += j_pr * \
                  (np.log(j_pr) -
                   np.log(p1) -
                   np.log(p2))
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
