"""Tests for moirais.fn.manov -- MANOVA."""

import numpy as np
from moirais.fn.manov import manova, manov
from moirais.fn._containers import DescriptiveResult


class TestManova:
    def test_alias(self):
        assert manov is manova

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(2, 1, (20, 3))])
        g = np.array([0]*20 + [1]*20)
        res = manova(X, g)
        assert isinstance(res, DescriptiveResult)

    def test_all_statistics_present(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(3, 1, (20, 3))])
        g = np.array([0]*20 + [1]*20)
        res = manova(X, g)
        assert "pillai" in res.extra
        assert "wilks" in res.extra
        assert "hotelling_lawley" in res.extra
        assert "roy" in res.extra

    def test_significant_difference(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(5, 0.5, (30, 2))])
        g = np.array([0]*30 + [1]*30)
        res = manova(X, g)
        assert res.extra["p_wilks"] < 0.05

    def test_wilks_bounded(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(1, 1, (20, 2))])
        g = np.array([0]*20 + [1]*20)
        res = manova(X, g)
        assert 0 <= res.value <= 1
