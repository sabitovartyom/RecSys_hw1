"""Weighted matrix factorization с собственными ALS-обновлениями."""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix


def least_squares(preferences, fixed, reg, alpha):
    """Обновить одну сторону факторов при фиксированной второй стороне.

    В CSR хранятся ВСЕ наблюдаемые пары, в том числе с таргетом 0.
    Для них confidence = 1 + alpha, для ненаблюдаемых пар confidence = 1.
    """
    n_rows = preferences.shape[0]
    factors = np.zeros((n_rows, fixed.shape[1]))
    base = fixed.T @ fixed + reg * np.eye(fixed.shape[1])
    for row in range(n_rows):
        start, end = preferences.indptr[row:row + 2]
        if start == end:
            continue
        vectors = fixed[preferences.indices[start:end]]
        targets = preferences.data[start:end]
        system = base + alpha * (vectors.T @ vectors)
        rhs = (1 + alpha) * (vectors.T @ targets)
        factors[row] = solve(system, rhs, assume_a="pos", check_finite=False)
    return factors


class IALS:
    def __init__(self, factors=32, reg=1.0, alpha=20.0, iterations=10, seed=42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def targets(self, data):
        return (data.rating.to_numpy() >= 4).astype(float)

    def fit(self, data, shape, item_features=None):
        preferences = csr_matrix(
            (self.targets(data), (data.user.to_numpy(), data.item.to_numpy())),
            shape=shape,
        )
        # Не удаляем нули: низкая оценка — наблюдаемое отрицательное предпочтение.
        transposed = preferences.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.01, (shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = least_squares(
                preferences, self.item_factors, self.reg, self.alpha
            )
            self.item_factors = least_squares(
                transposed, self.user_factors, self.reg, self.alpha
            )
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
