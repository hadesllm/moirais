"""Tests for moirais.fn.kmrl — kernel mean residual life."""

import numpy as np
import pytest

from moirais.fn.kmrl import kmrl


class TestKmrl:
    def test_exponential_constant_mrl(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(2.0, 1000)
        res = kmrl(data, x_eval=np.array([1.0, 2.0, 3.0]))
        assert res["mrl"][0] == pytest.approx(2.0, abs=0.5)
        assert res["mrl"][1] == pytest.approx(2.0, abs=0.5)

    def test_decreasing_for_bounded(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, 500)
        res = kmrl(data, x_eval=np.array([2.0, 5.0, 8.0]))
        assert res["mrl"][0] > res["mrl"][2]

    def test_output_keys(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = kmrl(data)
        assert "mrl" in res
        assert "x_eval" in res
        assert "bw" in res

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kmrl(np.array([1.0]))
