"""Дополнительный baseline, не заменяющий ни одну из пяти моделей."""

import numpy as np

from .common import interaction_matrix


class Popularity:
    def fit(self, data, shape, item_features=None):
        matrix = interaction_matrix(data, shape, positive_only=True)
        self.counts = np.asarray(matrix.sum(axis=0)).ravel()
        return self

    def score(self, users):
        return np.broadcast_to(self.counts, (len(users), len(self.counts))).copy()
