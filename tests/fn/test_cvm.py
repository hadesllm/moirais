"""Tests for cramer_von_mises_test."""
import numpy as np, pytest
from morie.fn.cvm import cramer_von_mises_test


class TestCramerVonMises:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = cramer_von_mises_test(x)
        assert r.test_name == "Cramer-von Mises test"
        assert r.p_value > 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            cramer_von_mises_test([1.0])
