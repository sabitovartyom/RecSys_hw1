"""EASE: closed-form решение линейной модели с нулевой диагональю."""

import numpy as np

from .common import interaction_matrix


class EASE:
    def __init__(self, reg=100.0):
        self.reg = reg

    def fit(self, data, shape, item_features=None):
        self.history = interaction_matrix(data, shape, positive_only=True)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::shape[1] + 1] += self.reg
        precision = np.linalg.inv(gram)
        # Нормировка по столбцам, поскольку предсказания равны X @ B.
        precision /= -np.diag(precision).copy()[None, :]
        np.fill_diagonal(precision, 0)
        self.weights = precision
        return self

    def score(self, users):
        return self.history[users] @ self.weights
