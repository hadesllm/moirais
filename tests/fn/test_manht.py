"""Tests for moirais.fn.manht — Manhattan plot data."""
import numpy as np
import pytest
from moirais.fn.manht import manhattan_data


class TestManhattan:
    def test_basic(self):
        pv = np.random.default_rng(42).uniform(1e-8, 1, 100)
        chrs = np.repeat([1, 2, 3, 4, 5], 20)
        pos = np.tile(np.arange(20) * 1000, 5)
        res = manhattan_data(pv, chrs, pos)
        assert res.extra["data"].shape[0] == 100

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            manhattan_data(np.array([0.1]), np.array([1, 2]), np.array([100]))
