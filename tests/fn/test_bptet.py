"""Tests for bowman_shenton_test."""
import numpy as np, pytest
from morie.fn.bptet import bowman_shenton_test


class TestBowmanShenton:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = bowman_shenton_test(x)
        assert r.test_name == "Bowman-Shenton test"
        assert r.p_value > 0.05

    def test_skewed_rejects(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1, 200)
        r = bowman_shenton_test(x)
        assert r.p_value < 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            bowman_shenton_test([1, 2, 3])
