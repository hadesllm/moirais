"""Tests for moirais.fn.cfanl -- Confirmatory factor analysis."""

import numpy as np
from moirais.fn.cfanl import cfa_uls, cfanl
from moirais.fn._containers import CfaRes


class TestCfaUls:
    def test_alias(self):
        assert cfanl is cfa_uls

    def test_returns_cfa_res(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 6))
        model = {"F1": [0, 1, 2], "F2": [3, 4, 5]}
        res = cfa_uls(X, model=model)
        assert isinstance(res, CfaRes)

    def test_fit_indices_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 6))
        model = {"F1": [0, 1, 2], "F2": [3, 4, 5]}
        res = cfa_uls(X, model=model)
        assert 0 <= res.cfi <= 1
        assert 0 <= res.tli <= 1
        assert res.rmsea >= 0
        assert res.srmr >= 0

    def test_loadings_dict(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 4))
        model = {"F1": [0, 1], "F2": [2, 3]}
        res = cfa_uls(X, model=model)
        assert "F1" in res.loadings
        assert "F2" in res.loadings
