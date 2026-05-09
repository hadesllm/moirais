"""Tests for moirais.fn.mcdir -- McDiarmid bounded-differences inequality."""

import numpy as np
import pytest
from moirais.fn.mcdir import mcdiarmid_bound


class TestMcDiarmid:
    def test_basic_bound(self):
        r = mcdiarmid_bound(100, 0.5)
        assert 0 < r["bound"] <= 1
        assert r["method"] == "McDiarmid inequality"

    def test_large_t_small_bound(self):
        r = mcdiarmid_bound(100, 50.0)
        assert r["bound"] < 1e-10

    def test_vector_c(self):
        c = np.array([0.5, 1.0, 0.5, 1.0, 0.5])
        r = mcdiarmid_bound(5, 1.0, c=c)
        assert r["sum_c_sq"] == pytest.approx(2.75)

    def test_bound_decreases_with_t(self):
        b1 = mcdiarmid_bound(50, 0.1)["bound"]
        b2 = mcdiarmid_bound(50, 1.0)["bound"]
        assert b2 < b1

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be"):
            mcdiarmid_bound(0, 0.5)

    def test_invalid_c(self):
        with pytest.raises(ValueError, match="constants must be"):
            mcdiarmid_bound(5, 0.5, c=-1.0)
