"""Tests for jarque_bera_test."""
import numpy as np, pytest
from morie.fn.jbtet import jarque_bera_test


class TestJarqueBera:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = jarque_bera_test(x)
        assert r.test_name == "Jarque-Bera test"
        assert r.p_value > 0.05

    def test_skewed_rejects(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1, 200)
        r = jarque_bera_test(x)
        assert r.p_value < 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            jarque_bera_test([1, 2, 3])
