"""Tests for normal_var_ratio_test."""
import numpy as np, pytest
from morie.fn.nvrmt import normal_var_ratio_test


class TestNormalVarRatio:
    def test_equal_variances(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        r = normal_var_ratio_test(x, y)
        assert r.test_name == "Variance ratio F-test"
        assert r.p_value > 0.05

    def test_unequal_variances(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 5, 50)
        r = normal_var_ratio_test(x, y)
        assert r.p_value < 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            normal_var_ratio_test([1.0], [2.0])
