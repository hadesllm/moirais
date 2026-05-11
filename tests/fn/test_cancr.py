"""Tests for morie.fn.cancr -- Canonical correlation analysis."""

import numpy as np
from morie.fn.cancr import canonical_correlation, cancr
from morie.fn._containers import DescriptiveResult


class TestCanonicalCorrelation:
    def test_alias(self):
        assert cancr is canonical_correlation

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        Y = rng.standard_normal((50, 2))
        res = canonical_correlation(X, Y)
        assert isinstance(res, DescriptiveResult)

    def test_correlations_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        Y = rng.standard_normal((50, 2))
        res = canonical_correlation(X, Y)
        corrs = res.value
        assert np.all(corrs >= -1e-10)
        assert np.all(corrs <= 1.0 + 1e-6)

    def test_n_components(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        Y = rng.standard_normal((50, 2))
        res = canonical_correlation(X, Y)
        assert len(res.value) == 2
        assert res.extra["X_scores"].shape == (50, 2)
