"""Tests for morie.fn.fa — Exploratory Factor Analysis."""

import numpy as np
import pytest
from morie.fn.fa import fa
from morie.fn._containers import FaRes


class TestFa:
    """Tests for Exploratory Factor Analysis."""

    def test_returns_fa_res(self, rng):
        """Two latent factors should be recoverable."""
        n = 300
        f1 = rng.standard_normal(n)
        f2 = rng.standard_normal(n)
        X = np.column_stack([
            f1 + rng.standard_normal(n) * 0.3,
            f1 + rng.standard_normal(n) * 0.3,
            f1 + rng.standard_normal(n) * 0.3,
            f2 + rng.standard_normal(n) * 0.3,
            f2 + rng.standard_normal(n) * 0.3,
            f2 + rng.standard_normal(n) * 0.3,
        ])
        result = fa(X, n_factors=2)
        assert isinstance(result, FaRes)
        assert result.loadings.shape == (6, 2)

    def test_communalities_range(self, rng):
        """Communalities should be in (0, 1)."""
        n = 200
        f = rng.standard_normal(n)
        X = np.column_stack([f + rng.standard_normal(n) * 0.4 for _ in range(5)])
        result = fa(X, n_factors=1)
        assert all(0 < h < 1 for h in result.communalities)

    def test_parallel_analysis_auto(self, rng):
        """When n_factors=None, parallel analysis should pick something."""
        n = 200
        f = rng.standard_normal(n)
        X = np.column_stack([f + rng.standard_normal(n) * 0.3 for _ in range(4)]
                           + [rng.standard_normal(n) for _ in range(2)])
        result = fa(X, n_factors=None)
        assert result.n_factors >= 1

    def test_no_rotation(self, rng):
        n = 200
        f = rng.standard_normal(n)
        X = np.column_stack([f + rng.standard_normal(n) * 0.3 for _ in range(4)])
        result = fa(X, n_factors=1, rotation="none")
        assert result.rotation == "none"
