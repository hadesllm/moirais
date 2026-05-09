"""Tests for sign_change_test."""
import numpy as np, pytest
from moirais.fn.sgchg import sign_change_test


class TestSignChange:
    def test_positive_shift(self):
        x = [5, 6, 7, 8, 9]
        r = sign_change_test(x, mu=0)
        assert r.test_name == "Sign test"
        assert r.p_value < 0.1

    def test_symmetric(self):
        x = [-2, -1, 1, 2]
        r = sign_change_test(x, mu=0)
        assert r.p_value > 0.5

    def test_empty_after_filter(self):
        with pytest.raises(ValueError):
            sign_change_test([5, 5, 5], mu=5)
