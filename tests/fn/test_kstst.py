"""Tests for ks_test."""
import numpy as np, pytest
from moirais.fn.kstst import ks_test


class TestKSTest:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = ks_test(x, distribution="norm")
        assert r.test_name == "Kolmogorov-Smirnov test"
        assert r.p_value > 0.05

    def test_two_sample(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 11, 12, 13, 14]
        r = ks_test(x, y)
        assert r.p_value < 0.05

    def test_empty(self):
        with pytest.raises(ValueError):
            ks_test([])
