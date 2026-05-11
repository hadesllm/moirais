"""Tests for morie.fn.kbwpi — Sheather-Jones plug-in bandwidth."""

import numpy as np
import pytest

from morie.fn.kbwpi import kbwpi


class TestKbwpi:
    def test_returns_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kbwpi(data)
        assert res["bw_opt"] > 0

    def test_normal_data_reasonable(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        res = kbwpi(data)
        assert 0.05 < res["bw_opt"] < 1.0

    def test_n_returned(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = kbwpi(data)
        assert res["n"] == 5

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbwpi(np.array([1.0]))
