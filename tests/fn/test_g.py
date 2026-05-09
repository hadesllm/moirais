"""Tests for moirais.fn.g -- Hedges' g effect size."""

import numpy as np
import pytest
from moirais.fn.g import hedges_g
from moirais.fn.d import cohens_d


class TestHedgesG:
    def test_smaller_than_d(self):
        """Hedges' g should be slightly smaller in magnitude than Cohen's d."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [6.0, 7.0, 8.0, 9.0, 10.0]
        g = hedges_g(x, y)
        d = cohens_d(x, y)
        assert isinstance(g, float)
        assert abs(g) < abs(d)

    def test_identical_groups_zero(self):
        """Identical groups give g = 0 (or NaN if pooled SD is 0)."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        g = hedges_g(x, y)
        assert g == pytest.approx(0.0, abs=1e-10)

    def test_large_n_converges_to_d(self):
        """With large N, correction factor J -> 1 so g -> d."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(1, 1, 500)
        g = hedges_g(x, y)
        d = cohens_d(x, y)
        assert abs(g - d) < 0.02
