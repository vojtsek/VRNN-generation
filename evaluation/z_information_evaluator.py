import os
import numpy as np
from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import wandb
from sklearn.feature_selection import mutual_info_classif

from .commons import Evaluator, TurnRecord, CorpusVocab

EPS=1e-15
def mutual_information_2d(x, y, sigma=1, normalized=False):
    bins = (256, 256)

    x = np.array([int(xx) for xx in x])
    y = np.array([int(xx) for xx in y])
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi


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
        return [f'{zi}-{tk}' for i, zi in enumerate(z_lst)]

    @staticmethod
    def _make_vec(z_str):
        return [f'{zi}' for i, zi in enumerate(z_str.split())]

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

        heatmap_m = np.eye(len(prior_vec), len(prior_vec)) * 1
        for z, zz in combinations(range(len(prior_vec)), 2):
            x = np.array([int(xx) for xx in self.z_vecs[z]])
            y = np.array([int(xx) for xx in self.z_vecs[zz]])
            mi = mutual_info_classif(x.reshape(-1, 1), y, discrete_features=True)[0]
            #mi = ZInfoEvaluator.compute_mi_two_vec(self.z_vecs[z], self.z_vecs[zz])
            heatmap_m[z, zz] = mi
            heatmap_m[zz, z] = mi
        fig = plt.figure(dpi=500, figsize=(10, 10))
        img = plt.imshow(heatmap_m, cmap='hot')
        #img = fig.figimage(heatmap_m, cmap='hot')
        plt.colorbar(img)
        wandb.log({'Z_MI': plt})
        plt.clf()
        plt.close()


    def compute_mi(self):
        mi = [0 for _ in range(len(self.prior_z_vocabs))]
#        print(self.joint_vocabs[0].vocab)
#        print(self.prior_z_vocabs[0].vocab)
        all_tks = []
        all_zs = [[] for _ in range(len(self.prior_z_vocabs))]
        for record in self.records:
            utt = record.gt_utterance.split()
            all_tks.extend(utt)
            for n, z in enumerate(record.prior_z_vector):
                all_zs[n].extend([z] * len(utt))
        for n in range(len(self.prior_z_vocabs)):
            x = np.array(all_tks).reshape(-1, 1)
            y = [int(xx[1]) for xx in all_zs[n]]
            mi[n] = mutual_info_classif(x, y, discrete_features=True)[0]
        return mi

    @staticmethod
    def compute_mi_two_vec(vec1, vec2):
        joint = Counter()
        pr1 = Counter()
        pr2 = Counter()

        for el1, el2 in zip(vec1, vec2):
            pr1.update([el1])
            pr2.update([el2])
            if el1 > el2:
                el2, el1 = el1, el2
            joint.update([f'[{el1}][{el2}]'])
        mi = 0
        for el1 in vec1:
            for el2 in vec2:
                p1 = pr1[el1] / sum(pr1.values()) + 1e-15
                p2 = pr2[el2] / sum(pr2.values()) + 1e-15
                if el1 > el2:
                    el2, el1 = el1, el2
                j_pr = joint[f'[{el1}][{el2}]'] / sum(joint.values()) + 1e-15
                print(f'joint: {j_pr},  p1: {p1}, p2: {p2}')
                mi += j_pr * np.log2(j_pr / (p1*p2))
                    #  (np.log2(j_pr) -
                    #   np.log2(p1) -
                    #   np.log2(p2))
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
