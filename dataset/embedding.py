import pickle
import random
import io

import numpy as np


class Embeddings:

    class MyDefDict:
        def __init__(self, dct):
            self.dct = dct

        def __getitem__(self, item):
            if item in self.dct:
                return self.dct[item]
            else:
                return self.dct[Embeddings.UNK]

    PAD = '<PAD>'
    EOS = '<EOS>'
    BOS = '<BOS>'
    UNK = '<UNK>'
    SPEC_TOKENS = [PAD, EOS, BOS, UNK]

    def __init__(self, data_fn, out_fn=None, distance='cos'):
        self.distance = distance
        self.id2w = {}
        self._w2id = {}
        try:
            with open(data_fn, 'rb') as inf:
                print('Loading embeddings from "{}"'.format(data_fn))
                self._data, self._w2id = pickle.load(inf)
                self.n, self.d = len(self._data), len(self._data[0])
        except pickle.UnpicklingError:
            with io.open(data_fn, 'r', encoding='utf-8', newline='\n', errors='ignore') as inf:
                self.n, self.d = map(int, inf.readline().split())
                print('Failed. Reading {} embeddings from "{}"'.format(self.n, data_fn))
                self._data = []
                print('Reading {} vectors of dimension {}'.format(self.n, self.d))
                for line in inf:
                    tokens = line.rstrip().split(' ')
                    self._data.append(list(map(float, tokens[1:])))
                    self._w2id[tokens[0]] = len(self._w2id)
                for special_tk in Embeddings.SPEC_TOKENS:
                    self._data.append([random.gauss(0, 1) for _ in range(self.d)])
                    self._w2id[special_tk] = len(self._w2id)
            if out_fn is not None:
                with open(out_fn, 'wb') as of:
                    pickle.dump((self._data, self._w2id), of)
        self.id2w = {y: x for x, y in self._w2id.items()}
        self.w2id = Embeddings.MyDefDict(self._w2id)

    def __getitem__(self, key):
        idx = self.w2id[key] if isinstance(key, str) else key
        return self._data[idx]

    def embed_tokens(self, tokens, token_weights=None):
        if token_weights is None:
            token_weights = np.ones((len(tokens),))
        tokens = np.array([self[tk] for tk in tokens]).T
        res = np.matmul(tokens, token_weights) / len(tokens)
        return res

    def embedding_similarity(self, e1, e2, distance=None):
        distance = self.distance if distance is None else distance
        if distance == 'l1':
            return 1 / np.linalg.norm(e1 - e2, ord=1)
        elif distance == 'l2':
            return 1 / np.linalg.norm(e1 - e2, ord=2)
        else:
            cosf = np.dot(e1, e2) / (np.linalg.norm(e1, ord=2) * np.linalg.norm(e2, ord=2))
            return cosf
