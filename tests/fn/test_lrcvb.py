"""Test learning_curve_bio (lrcvb)."""
import numpy as np
from morie.fn.lrcvb import learning_curve_bio, lrcvb
from morie.fn._containers import DescriptiveResult


class TestLrcvb:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 2, rng.standard_normal((30, 3)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = learning_curve_bio(X, y, cv=3)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "learning_curve_bio"
        assert len(result.extra["val_scores"]) == 5

    def test_monotone_train(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 3)) + 3, rng.standard_normal((30, 3)) - 3])
        y = np.array([1] * 30 + [0] * 30)
        result = learning_curve_bio(X, y, cv=3)
        assert result.extra["train_scores"][-1] >= result.extra["train_scores"][0] - 0.1

    def test_alias(self):
        assert lrcvb is learning_curve_bio
