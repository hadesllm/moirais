"""Tests for em_imputation."""
import numpy as np, pytest
from moirais.fn.emimq import em_imputation

class TestEM:
    def test_basic(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (30, 3))
        data[0, 0] = np.nan
        data[5, 2] = np.nan
        r = em_imputation(data)
        assert r.extra["n_missing"] == 2

    def test_means_reasonable(self):
        rng = np.random.default_rng(1)
        data = rng.normal(5, 1, (50, 2))
        data[0, 0] = np.nan
        r = em_imputation(data)
        assert all(3 < m < 7 for m in r.extra["imputed_means"])
