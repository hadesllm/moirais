"""Tests for moirais.fn.fanlz -- Factor analysis (ML)."""

import numpy as np
from moirais.fn.fanlz import factor_analysis_ml, fanlz
from moirais.fn._containers import FaRes


class TestFactorAnalysisMl:
    def test_alias(self):
        assert fanlz is factor_analysis_ml

    def test_returns_fa_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 6))
        res = factor_analysis_ml(X, n_factors=2)
        assert isinstance(res, FaRes)

    def test_loadings_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 6))
        res = factor_analysis_ml(X, n_factors=2)
        assert res.loadings.shape == (6, 2)
        assert len(res.communalities) == 6

    def test_communalities_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        res = factor_analysis_ml(X, n_factors=2)
        assert np.all(res.communalities >= 0)

    def test_rotation_none(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        res = factor_analysis_ml(X, n_factors=2)
        assert res.rotation == "none"
