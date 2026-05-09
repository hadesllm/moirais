"""Tests for cfaes -- ESEM."""
import numpy as np
from moirais.fn.cfaes import cfa_esem
from moirais.fn._containers import DescriptiveResult


class TestCfaEsem:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 10))
        result = cfa_esem(X, n_factors=2)
        assert isinstance(result, DescriptiveResult)
        assert "loadings" in result.value

    def test_srmr_computed(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, (100, 8))
        result = cfa_esem(X, n_factors=2)
        assert "srmr" in result.value
        assert result.value["srmr"] >= 0
