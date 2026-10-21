"""Тот же ALS и confidence, но таргет учитывает оценку и давность."""

import numpy as np

from .ials import IALS


class CustomIALS(IALS):
    def __init__(
        self, factors=32, reg=1.0, alpha=20.0, iterations=10, seed=42,
        use_rating=True, half_life_days=365,
    ):
        super().__init__(factors, reg, alpha, iterations, seed)
        self.use_rating = use_rating
        self.half_life_days = half_life_days

    def targets(self, data):
        targets = super().targets(data)
        if self.use_rating:
            targets *= np.clip((data.rating.to_numpy() - 3) / 2, 0, 1)
        if self.half_life_days is not None:
            # Конец именно текущей обучающей истории, без дат из validation/test.
            age_days = (data.timestamp.max() - data.timestamp).dt.total_seconds() / 86400
            targets *= np.exp2(-age_days.to_numpy() / self.half_life_days)
        return targets
