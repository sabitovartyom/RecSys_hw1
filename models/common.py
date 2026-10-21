"""Матрицы взаимодействий, общие для моделей и оценки."""

import numpy as np
from scipy.sparse import csr_matrix


def interaction_matrix(data, shape, positive_only=False):
    if positive_only:
        data = data[data.rating >= 4]
    matrix = csr_matrix(
        (np.ones(len(data)), (data.user.to_numpy(), data.item.to_numpy())),
        shape=shape,
    )
    matrix.data[:] = 1.0
    return matrix
