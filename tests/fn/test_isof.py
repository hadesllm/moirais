"""Tests for moirais.fn.isof -- Isolation Forest."""

import numpy as np
from moirais.fn.isof import isolation_forest, isof
from moirais.fn._containers import DescriptiveResult


class TestIsof:
    def test_alias(self):
        assert isof is isolation_forest

    def test_outliers_higher_score(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (98, 2))
        outliers = np.array([[10.0, 10.0], [-10.0, -10.0]])
        X = np.vstack([normal, outliers])
        result = isolation_forest(X, n_trees=50, sample_size=50)
        assert isinstance(result, DescriptiveResult)
        scores = result.extra["scores"]
        assert scores[-1] > np.median(scores[:98])
        assert scores[-2] > np.median(scores[:98])

    def test_score_range(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        result = isolation_forest(X, n_trees=20)
        scores = result.extra["scores"]
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
