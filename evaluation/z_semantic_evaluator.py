import os
import copy
from itertools import groupby
from collections import Counter
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from .commons import Evaluator, TurnRecord


class ZSemanticEvaluator(Evaluator):

    def __init__(self, fn, fn_test):
        self.fn = fn
        self.fn_test = fn_test

    def eval_from_dir(self, directory, role=None):
        fn = os.path.join(directory, self.fn)
        records = []
        slot_map = dict()
        TurnRecord.parse(fn, records, slot_map, role)
        test_records = []
        fn = os.path.join(directory, self.fn_test)
        TurnRecord.parse(fn, test_records, {}, role)
        if role == 'system':
            def _oh(idx, size):
                oh = [0] * size
                oh[idx] = 1
                return oh

            dt_clf = tree.DecisionTreeClassifier(max_depth=7, criterion='gini')
            rf_clf = RandomForestClassifier()
            X = []
            y = []
            classes = []
            records = sorted(records, key=lambda r: r.turn_type)
            for cls, (t_tpe, records) in enumerate(groupby(records, key=lambda r: r.turn_type)):
                r = copy.deepcopy(records)
                print(t_tpe, len(list(r)))
                classes.append(t_tpe)
                t_counter = Counter()
                for record in records:
                    d = []
                    for i in record.prior_z_vector:
                        d.extend(_oh(i[1], 20))
                    X.append(d)
                    y.append(t_tpe)
                    t_counter.update([str(i) for i in record.prior_z_vector])
                print(t_counter.most_common(5))
            dt_clf.fit(X, y)
            rf_clf.fit(X, y)
            X_test, y_test = [], []
            test_records = sorted(test_records, key=lambda r: r.turn_type)
            for cls, (t_tpe, records) in enumerate(groupby(test_records, key=lambda r: r.turn_type)):
                r = copy.deepcopy(records)
                for record in records:
                    d = []
                    for i in record.prior_z_vector:
                        d.extend(_oh(i[1], 20))
                    X_test.append(d)
                    y_test.append(t_tpe)

            y_hat = dt_clf.predict(X)
            y_hat_rf = rf_clf.predict(X)
            acc = accuracy_score(y, y_hat)

            print('DT accuracy:', acc)
            fig = plt.gcf()
            fig.set_size_inches(15, 9)
            tree.plot_tree(dt_clf,
                           filled=True,
                           label='none',
                           proportion=True,
                           max_depth=8,
                           class_names=classes,
                           fontsize=10)
            plt.savefig(os.path.join(directory, f'dt.png'), format='png', bbox_inches="tight")
        else:
            for val, zs in slot_map.items():
                print(val, zs.most_common(5))
