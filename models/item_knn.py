"""Item-based CF: cosine similarity и k соседей для каждого товара истории."""

import numpy as np

from .common import interaction_matrix


class ItemKNN:
    def __init__(self, k=100, shrink=10.0):
        self.k = k
        self.shrink = shrink

    def fit(self, data, shape, item_features=None):
        self.history = interaction_matrix(data, shape, positive_only=True)
        similarity = (self.history.T @ self.history).tocsr()
        norms = np.sqrt(self.history.power(2).sum(axis=0)).A1
        rows = np.repeat(np.arange(shape[1]), np.diff(similarity.indptr))
        similarity.data /= norms[rows] * norms[similarity.indices] + self.shrink
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        # Каждый товар из истории голосует только за k наиболее похожих товаров.
        for item in range(shape[1]):
            start, end = similarity.indptr[item:item + 2]
            values = similarity.data[start:end]
            if len(values) > self.k:
                discard = np.argpartition(values, -self.k)[:-self.k]
                values[discard] = 0
        similarity.eliminate_zeros()
        self.similarity = similarity
        return self

    def score(self, users):
        return (self.history[users] @ self.similarity).toarray()
