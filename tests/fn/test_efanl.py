"""Tests for moirais.fn.efanl -- Exploratory factor analysis (PAF)."""

import numpy as np
from moirais.fn.efanl import efa_principal_axis, efanl
from moirais.fn._containers import FaRes


class TestEfaPrincipalAxis:
    def test_alias(self):
        assert efanl is efa_principal_axis

    def test_returns_fa_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 6))
        res = efa_principal_axis(X, n_factors=2)
        assert isinstance(res, FaRes)

    def test_loadings_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 6))
        res = efa_principal_axis(X, n_factors=3)
        assert res.loadings.shape == (6, 3)

    def test_communalities_valid(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        res = efa_principal_axis(X, n_factors=2)
        assert np.all(res.communalities >= 0)
        assert np.all(res.communalities <= 1.0 + 1e-6)
