"""FM второго порядка на user/item ID и категориальных метапризнаках."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import interaction_matrix


def sample_unseen(users, seen, rng):
    """По одному ненаблюдаемому товару для каждого переданного пользователя."""
    # У пользователя с полным каталогом нельзя получить отрицательный пример.
    users = users[np.diff(seen.indptr)[users] < seen.shape[1]]
    items = rng.integers(seen.shape[1], size=len(users))
    if len(users) == 0:
        return users, items
    observed = np.asarray(seen[users, items]).ravel() != 0
    while observed.any():
        items[observed] = rng.integers(seen.shape[1], size=observed.sum())
        observed = np.asarray(seen[users, items]).ravel() != 0
    return users, items


class FM(nn.Module):
    def __init__(
        self, factors=32, lr=0.005, reg=1e-5, epochs=5,
        batch_size=2048, negative_ratio=2, use_metadata=True, device="cpu", seed=42,
    ):
        super().__init__()
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.use_metadata = use_metadata
        self.device_name = device
        self.seed = seed

    def forward(self, features):
        vectors = self.embedding(features)
        linear = self.bias + self.linear(features).sum(dim=1).squeeze(-1)
        interactions = 0.5 * (
            vectors.sum(dim=1).square() - vectors.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interactions

    def fit(self, data, shape, item_features=None):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        n_users, n_items = shape
        if self.use_metadata:
            self.item_features = item_features
        else:
            self.item_features = (np.arange(n_items) + n_users)[:, None]
        n_features = int(self.item_features.max()) + 1
        self.embedding = nn.Embedding(n_features, self.factors)
        self.linear = nn.Embedding(n_features, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        self.to(self.device_name)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        users = data.user.to_numpy()
        items = data.item.to_numpy()
        labels = (data.rating.to_numpy() >= 4).astype(np.float32)
        seen = interaction_matrix(data, shape)
        positive_users = np.repeat(users[labels == 1], self.negative_ratio)
        self.loss_history = []

        for _ in range(self.epochs):
            # Отрицательные примеры исключают только известную обучающую историю.
            negative_users, negative_items = sample_unseen(positive_users, seen, rng)
            epoch_users = np.concatenate([users, negative_users])
            epoch_items = np.concatenate([items, negative_items])
            epoch_labels = np.concatenate([labels, np.zeros(len(negative_users), dtype=np.float32)])
            features = np.column_stack([epoch_users, self.item_features[epoch_items]])
            features = torch.as_tensor(features, dtype=torch.long, device=self.device_name)
            target = torch.as_tensor(epoch_labels, device=self.device_name)
            order = rng.permutation(len(features))
            total_loss = 0.0

            for start in range(0, len(order), self.batch_size):
                batch = torch.as_tensor(
                    order[start:start + self.batch_size], device=self.device_name
                )
                loss = F.binary_cross_entropy_with_logits(self(features[batch]), target[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.loss_history.append(total_loss / len(features))

        # Кеширование точной FM-формулы: all-item ranking без тензора user × item × field.
        self.cpu()
        self.eval()
        with torch.no_grad():
            item_fields = torch.as_tensor(self.item_features, dtype=torch.long)
            vectors = self.embedding(item_fields)
            self.item_vectors = vectors.sum(dim=1).numpy()
            self.item_biases = (
                self.linear(item_fields).sum(dim=1).squeeze(-1)
                + 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1)).sum(dim=1)
            ).numpy()
            self.user_vectors = self.embedding.weight[:n_users].numpy()
            self.user_biases = self.linear.weight[:n_users, 0].numpy() + self.bias.item()
        return self

    def score(self, users):
        return (
            self.user_vectors[users] @ self.item_vectors.T
            + self.item_biases[None, :] + self.user_biases[users, None]
        )
