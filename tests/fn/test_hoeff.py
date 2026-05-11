"""Tests for morie.fn.hoeff -- Hoeffding concentration inequality."""

import numpy as np
import pytest
from morie.fn.hoeff import hoeffding_bound


class TestHoeffding:
    def test_basic_bound(self):
        r = hoeffding_bound(100, 0.1)
        assert 0 < r["bound"] <= 1
        assert r["method"] == "Hoeffding inequality"

    def test_large_t_gives_small_bound(self):
        r = hoeffding_bound(100, 1.0)
        assert r["bound"] < 1e-50

    def test_small_t_gives_large_bound(self):
        r = hoeffding_bound(10, 0.01)
        assert r["bound"] > 0.99

    def test_bound_decreases_with_n(self):
        b1 = hoeffding_bound(50, 0.1)["bound"]
        b2 = hoeffding_bound(500, 0.1)["bound"]
        assert b2 < b1

    def test_wider_range_loosens_bound(self):
        b1 = hoeffding_bound(100, 0.1, a=0, b=1)["bound"]
        b2 = hoeffding_bound(100, 0.1, a=0, b=10)["bound"]
        assert b2 > b1

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be"):
            hoeffding_bound(0, 0.1)

    def test_invalid_t(self):
        with pytest.raises(ValueError, match="t must be"):
            hoeffding_bound(100, -1)

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="a must be"):
            hoeffding_bound(100, 0.1, a=5, b=3)
