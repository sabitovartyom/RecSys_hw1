"""Один протокол рекомендаций и метрик для всех моделей."""

import numpy as np
import pandas as pd

from models.common import interaction_matrix


def prepare_evaluation(history, future, shape, k=10):
    seen = interaction_matrix(history, shape)
    candidates = np.sort(history.item.unique())
    relevant = future[
        (future.rating >= 4)
        & future.user.isin(history.user.unique())
        & future.item.isin(candidates)
    ].copy()
    if relevant.empty:
        return seen, candidates, {}
    known = np.asarray(seen[relevant.user.to_numpy(), relevant.item.to_numpy()]).ravel()
    relevant = relevant[known == 0]
    counts = np.asarray(seen[:, candidates].getnnz(axis=1)).ravel()
    relevant = relevant[len(candidates) - counts[relevant.user.to_numpy()] >= k]
    truth = relevant.groupby("user").item.apply(lambda x: np.sort(x.unique())).to_dict()
    return seen, candidates, truth


def recommend(model, users, seen, candidates, k=10):
    scores = model.score(np.asarray(users))
    allowed = np.zeros(scores.shape[1], dtype=bool)
    allowed[candidates] = True
    scores[:, ~allowed] = -np.inf
    rows, columns = seen[users].nonzero()
    scores[rows, columns] = -np.inf
    # При равных score выбираем меньший item ID: результат воспроизводим.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def ranking_metrics(recommendations, truth, k=10):
    hits = np.isin(recommendations[:k], truth).astype(float)
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = discounts[:min(len(truth), k)].sum()
    return {
        f"NDCG@{k}": float((hits * discounts).sum() / ideal),
        f"Recall@{k}": float(hits.sum() / len(truth)),
    }


def evaluate(model, seen, candidates, truth, k=10, batch_size=128):
    assert truth, "Нет пользователей для оценки: проверьте разбиение и фильтры."
    users = np.array(sorted(truth), dtype=np.int64)
    rows = []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        predictions = recommend(model, batch, seen, candidates, k)
        for user, prediction in zip(batch, predictions):
            rows.append({"user": user, **ranking_metrics(prediction, truth[user], k)})
    per_user = pd.DataFrame(rows).set_index("user")
    return per_user.mean().to_dict(), per_user
