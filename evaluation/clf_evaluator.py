import os
import copy
from itertools import groupby
from collections import Counter
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from .commons import Evaluator, TurnRecord


class ClfEvaluator(Evaluator):

    def __init__(self, fn, fn_test):
        self.fn = fn
        self.fn_test = fn_test

    def _label(self, tpe):
        tpe = tpe.split('-')[0]
 #       if tpe.startswith('general'):
#            tpe = 'goodbye'
        return tpe

    def eval_from_dir(self, directory, role=None):
        fn = os.path.join(directory, self.fn)
        records = []
        slot_map = dict()
        TurnRecord.parse(fn, records, slot_map, role)
        test_records = []
        fn = os.path.join(directory, self.fn_test)
        TurnRecord.parse(fn, test_records, {}, role)
        def _oh(idx, size):
            oh = [0] * size
            oh[idx] = 1
            return oh

        clf = LogisticRegression()
        X = []
        y = []
        classes = []
        all_y_classes = Counter()
        records = sorted(records, key=lambda r: r.turn_type)
        for cls, (t_tpe, records) in enumerate(groupby(records, key=lambda r: r.turn_type)):
            r = copy.deepcopy(records)
            t_tpe = self._label(t_tpe)
            if t_tpe in  ['unk', '']:
                continue
            # print(t_tpe, len(list(r)))
            classes.append(t_tpe)
            t_counter = Counter()
            for record in records:
                X.append([r[1]/20 for r in record.prior_z_vector])
                y.append(t_tpe)
                all_y_classes.update([t_tpe])
                t_counter.update([str(i) for i in record.posterior_z_vector])
            print(t_counter.most_common(5))
        clf.fit(X, y)
        X_test, y_test = [], []
        most_common = all_y_classes.most_common(1)[0][0]
        test_records = sorted(test_records, key=lambda r: r.turn_type)
        classes = sorted(list(set(classes)))
        for cls, (t_tpe, records) in enumerate(groupby(test_records, key=lambda r: r.turn_type)):
            r = copy.deepcopy(records)
            if t_tpe == 'unk':
                continue
            for record in records:
                X_test.append([r[1]/20 for r in record.posterior_z_vector])
                t_tpe = self._label(t_tpe)
                y_test.append(t_tpe)

        y_hat = clf.predict(X_test)
        acc = accuracy_score(y_test, y_hat)

        print('DT accuracy:', acc)
        print('majority:', accuracy_score(y_test, [most_common] * len(y_test)))
