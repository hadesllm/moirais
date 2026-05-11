"""Tests for morie.fn.kqant — kernel quantile estimator."""

import numpy as np
import pytest

from morie.fn.kqant import kqant


class TestKqant:
    def test_median_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5, 1, 500)
        res = kqant(data, probs=np.array([0.5]))
        assert res["quantiles"][0] == pytest.approx(5.0, abs=0.2)

    def test_quartiles_ordered(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 300)
        res = kqant(data)
        assert res["quantiles"][0] < res["quantiles"][1] < res["quantiles"][2]

    def test_custom_probs(self):
        data = np.arange(1, 101, dtype=float)
        res = kqant(data, probs=np.array([0.1, 0.9]))
        assert len(res["quantiles"]) == 2
        assert res["quantiles"][0] < res["quantiles"][1]

    def test_raises_invalid_probs(self):
        with pytest.raises(ValueError):
            kqant(np.array([1.0, 2.0, 3.0]), probs=np.array([0.0]))

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kqant(np.array([1.0]))
