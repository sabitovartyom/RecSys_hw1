"""Быстрые проверки на маленьких данных: без загрузки Amazon и GPU."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from data import clean_reviews, encode_ids, make_item_features, read_metadata, read_reviews, temporal_split
from evaluation import evaluate, prepare_evaluation, ranking_metrics, recommend
from experiment import final_comparison, tune_models
from models import CustomIALS, EASE, FM, IALS, ItemKNN, Popularity
from models.common import interaction_matrix
from models.fm import sample_unseen
from models.ials import least_squares


torch.set_num_threads(1)


def small_data():
    """24 пользователя, 48 товаров, у каждого есть прошлое и будущие оценки."""
    rows = []
    for user in range(24):
        for step in range(12):
            rows.append({
                "user": user,
                "item": (2 * user + step) % 48,
                "rating": 2 if step % 5 == 0 else 4 + step % 2,
                "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=step),
            })
    data = pd.DataFrame(rows)
    train = data[data.timestamp < pd.Timestamp("2020-01-09", tz="UTC")].copy()
    valid = data[(data.timestamp >= pd.Timestamp("2020-01-09", tz="UTC"))
                 & (data.timestamp < pd.Timestamp("2020-01-11", tz="UTC"))].copy()
    test = data[data.timestamp >= pd.Timestamp("2020-01-11", tz="UTC")].copy()
    items = pd.DataFrame({
        "brand": [f"brand_{i % 3}" for i in range(48)],
        "category": [f"category_{i % 4}" for i in range(48)],
    })
    return train, valid, test, items, (24, 48)


class DataTests(unittest.TestCase):
    def test_loading_cleaning_and_metadata(self):
        reviews = [
            {"reviewerID": "u", "asin": "a", "overall": 5, "unixReviewTime": 200},
            {"reviewerID": "u", "asin": "a", "overall": 2, "unixReviewTime": 100},
            {"reviewerID": "v", "asin": "b", "overall": 4, "unixReviewTime": 300},
            {"reviewerID": "v", "asin": "c", "overall": 9, "unixReviewTime": 400},
            {"reviewerID": None, "asin": "d", "overall": 4, "unixReviewTime": 500},
        ]
        metadata = [
            {"asin": "a", "brand": "A &amp; B", "category": ["root", "leaf"]},
            {"asin": "a", "brand": "duplicate"},
            {"asin": "b", "categories": [["root", "other"]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            review_path = Path(directory) / "reviews.json.gz"
            meta_path = Path(directory) / "metadata.json.gz"
            for path, rows in [(review_path, reviews), (meta_path, metadata)]:
                with gzip.open(path, "wt", encoding="utf-8") as file:
                    for row in rows:
                        file.write(json.dumps(row) + "\n")
            data, users, items = encode_ids(clean_reviews(read_reviews(review_path)))
            self.assertEqual(len(data), 2)
            self.assertEqual(data.iloc[0].rating, 2)
            self.assertEqual(len(users), 2)
            meta = read_metadata(meta_path, [*items, "missing"])
            self.assertEqual(meta.iloc[0].brand, "A & B")
            self.assertEqual(meta.iloc[0].category, "leaf")
            self.assertEqual(meta.iloc[1].category, "other")
            self.assertEqual(meta.iloc[2].brand, "")

    def test_time_split_keeps_equal_timestamps_together(self):
        train, valid, test, _, _ = small_data()
        parts = temporal_split(pd.concat([train, valid, test]))
        self.assertLess(parts[0].timestamp.max(), parts[1].timestamp.min())
        self.assertLess(parts[1].timestamp.max(), parts[2].timestamp.min())

    def test_metadata_vocabulary_uses_train_only(self):
        items = pd.DataFrame({"brand": ["a", "future", ""], "category": ["x", "new", ""]})
        train = pd.DataFrame({"item": [0]})
        features = make_item_features(items, train, n_users=2)
        np.testing.assert_array_equal(features[:, 0], [2, 3, 4])
        # Будущие и отсутствующие значения попадают в один unknown-код.
        np.testing.assert_array_equal(features[1, 1:], features[2, 1:])
        self.assertGreater(features[:, 1].min(), features[:, 0].max())
        self.assertGreater(features[:, 2].min(), features[:, 1].max())


class FormulaTests(unittest.TestCase):
    def test_als_update_matches_dense_normal_equations(self):
        preferences = csr_matrix(([1.0, 0.0, 0.5], ([0, 0, 1], [0, 2, 1])), shape=(2, 4))
        fixed = np.random.default_rng(2).normal(size=(4, 3))
        actual = least_squares(preferences, fixed, reg=0.7, alpha=4)
        dense = preferences.toarray()
        for user in range(2):
            confidence = np.ones(4)
            observed = preferences[user].indices
            confidence[observed] = 5
            expected = np.linalg.solve(
                fixed.T @ (confidence[:, None] * fixed) + 0.7 * np.eye(3),
                fixed.T @ (confidence * dense[user]),
            )
            np.testing.assert_allclose(actual[user], expected)

    def test_ease_solves_constrained_ridge(self):
        train, _, _, _, shape = small_data()
        model = EASE(reg=2).fit(train, shape)
        matrix = interaction_matrix(train, shape, positive_only=True).toarray()
        gram = matrix.T @ matrix
        for item in [0, 5, 20]:
            other = np.arange(shape[1]) != item
            expected = np.linalg.solve(
                gram[np.ix_(other, other)] + 2 * np.eye(other.sum()), gram[other, item]
            )
            np.testing.assert_allclose(model.weights[other, item], expected, atol=1e-10)
        np.testing.assert_array_equal(np.diag(model.weights), 0)

    def test_knn_matches_cosine_without_pruning(self):
        train, _, _, _, shape = small_data()
        model = ItemKNN(k=48, shrink=0).fit(train, shape)
        matrix = interaction_matrix(train, shape, positive_only=True).toarray()
        norms = np.linalg.norm(matrix, axis=0)
        expected = matrix.T @ matrix / (norms[:, None] * norms[None, :])
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.similarity.toarray(), expected)
        pruned = ItemKNN(k=3).fit(train, shape)
        self.assertTrue((np.diff(pruned.similarity.indptr) <= 3).all())

    def test_custom_targets_and_baseline_equivalence(self):
        train, _, _, _, shape = small_data()
        expected = (train.rating.to_numpy() >= 4).astype(float)
        np.testing.assert_array_equal(IALS().targets(train), expected)
        plain = CustomIALS(use_rating=False, half_life_days=None)
        np.testing.assert_array_equal(plain.targets(train), expected)
        actual = CustomIALS(half_life_days=2).targets(train)
        age = (train.timestamp.max() - train.timestamp).dt.total_seconds().to_numpy() / 86400
        expected *= np.clip((train.rating.to_numpy() - 3) / 2, 0, 1) * 2 ** (-age / 2)
        np.testing.assert_allclose(actual, expected)
        a = IALS(factors=4, iterations=2).fit(train, shape)
        b = CustomIALS(factors=4, iterations=2, use_rating=False, half_life_days=None).fit(train, shape)
        np.testing.assert_allclose(a.score([0, 1]), b.score([0, 1]))

    def test_fm_cached_scores_equal_pairwise_formula(self):
        train, _, _, items, shape = small_data()
        features = make_item_features(items, train, shape[0])
        model = FM(factors=4, epochs=2, batch_size=64).fit(train, shape, features)
        pairs = np.column_stack([np.repeat([0, 3], 48), np.tile(features, (2, 1))])
        with torch.no_grad():
            expected = model(torch.as_tensor(pairs)).numpy().reshape(2, 48)
        np.testing.assert_allclose(model.score(np.array([0, 3])), expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(np.isfinite(model.loss_history).all())


class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        actual = ranking_metrics(np.arange(10), np.array([0, 2, 20]))
        ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
        self.assertAlmostEqual(actual["NDCG@10"], (1 + 1 / np.log2(4)) / ideal)
        self.assertAlmostEqual(actual["Recall@10"], 2 / 3)

    def test_filter_cold_entities_and_mask_all_known_items(self):
        train, valid, _, _, shape = small_data()
        future = valid.copy()
        future.loc[len(future)] = [24, 0, 5, valid.timestamp.max()]
        future.loc[len(future)] = [0, 48, 5, valid.timestamp.max()]
        extended_shape = (25, 49)
        seen, candidates, truth = prepare_evaluation(train, future, extended_shape)
        self.assertNotIn(24, truth)
        self.assertNotIn(48, candidates)
        model = Popularity().fit(train, extended_shape)
        users = np.array(sorted(truth))
        recommendations = recommend(model, users, seen, candidates)
        for user, prediction in zip(users, recommendations):
            self.assertTrue(np.isin(prediction, candidates).all())
            self.assertEqual(seen[user, prediction].nnz, 0)
            self.assertEqual(len(np.unique(prediction)), 10)

    def test_negative_samples_are_unseen(self):
        train, _, _, _, shape = small_data()
        seen = interaction_matrix(train, shape)
        users, items = sample_unseen(np.repeat(np.arange(24), 4), seen, np.random.default_rng(3))
        self.assertEqual(np.asarray(seen[users, items]).sum(), 0)
        full = csr_matrix(np.ones((1, 3)))
        users, items = sample_unseen(np.array([0]), full, np.random.default_rng(3))
        self.assertEqual(len(users), 0)
        self.assertEqual(len(items), 0)

    def test_small_end_to_end_all_models(self):
        train, valid, test, items, shape = small_data()
        features = make_item_features(items, train, shape[0])
        specs = {
            "Popularity": (Popularity, [{}]),
            "ItemKNN": (ItemKNN, [{"k": 3}, {"k": 5}]),
            "iALS": (IALS, [{"factors": 4, "iterations": 2}]),
            "EASE": (EASE, [{"reg": 2}]),
            "FM": (FM, [{"factors": 4, "epochs": 1, "batch_size": 64}]),
            "custom iALS": (CustomIALS, [{"factors": 4, "iterations": 2}]),
            "FM IDs": (FM, [{"factors": 4, "epochs": 1, "use_metadata": False}]),
        }
        validation, best = tune_models(specs, train, valid, shape, features)
        self.assertEqual(len(validation), 8)
        self.assertEqual(set(best), set(specs))
        train_valid = pd.concat([train, valid], ignore_index=True)
        features = make_item_features(items, train_valid, shape[0])
        final, per_user = final_comparison(specs, best, train_valid, test, shape, features)
        self.assertEqual(len(final), 7)
        for metric in ["NDCG@10", "Recall@10"]:
            self.assertTrue(final[metric].between(0, 1).all())
        first = next(iter(per_user.values())).index
        for frame in per_user.values():
            self.assertTrue(frame.index.equals(first))


if __name__ == "__main__":
    unittest.main()
