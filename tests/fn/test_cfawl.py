"""Tests for cfawl -- WLSMV CFA."""
import numpy as np
from moirais.fn.cfawl import cfa_wlsmv
from moirais.fn._containers import DescriptiveResult


class TestCfaWlsmv:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 10))
        R = np.corrcoef(X, rowvar=False)
        spec = {"F1": [0, 1, 2, 3, 4], "F2": [5, 6, 7, 8, 9]}
        result = cfa_wlsmv(R, spec, n_obs=100)
        assert isinstance(result, DescriptiveResult)
        assert "srmr" in result.value

    def test_rmsea_nonneg(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 8))
        R = np.corrcoef(X, rowvar=False)
        spec = {"F1": [0, 1, 2, 3], "F2": [4, 5, 6, 7]}
        result = cfa_wlsmv(R, spec, n_obs=100)
        assert result.value["rmsea"] >= 0
