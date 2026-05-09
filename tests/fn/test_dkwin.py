"""Tests for moirais.fn.dkwin -- DKW inequality test."""

import numpy as np
import pytest
from moirais.fn.dkwin import dkw_test


class TestDKW:
    def test_normal_data_norm_cdf(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        r = dkw_test(x, cdf="norm")
        assert not r["reject"]
        assert r["D_n"] < r["critical_value"]

    def test_uniform_data_norm_cdf_rejects(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, 200)
        r = dkw_test(x, cdf="norm")
        assert r["reject"]

    def test_critical_value_formula(self):
        r = dkw_test(np.array([0.0, 1.0, 2.0]), cdf="norm", alpha=0.05)
        expected = np.sqrt(np.log(2.0 / 0.05) / 6.0)
        assert r["critical_value"] == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            dkw_test(np.array([]))

    def test_unknown_cdf(self):
        with pytest.raises(ValueError, match="Unknown"):
            dkw_test(np.array([1, 2, 3]), cdf="nonexistent")
