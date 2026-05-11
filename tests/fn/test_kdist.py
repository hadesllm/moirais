"""Tests for morie.fn.kdist — kernel CDF estimator."""

import numpy as np
import pytest

from morie.fn.kdist import kdist


class TestKdist:
    def test_monotone_increasing(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kdist(data)
        assert np.all(np.diff(res["cdf"]) >= -1e-12)

    def test_cdf_range(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kdist(data)
        assert res["cdf"][0] < 0.05
        assert res["cdf"][-1] > 0.95

    def test_median_near_half(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        res = kdist(data, x_eval=np.array([5.5]))
        assert res["cdf"][0] == pytest.approx(0.5, abs=0.15)

    def test_custom_bw(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        res = kdist(data, bw=0.5)
        assert res["bw"] == 0.5

    def test_raises_small_sample(self):
        with pytest.raises(ValueError):
            kdist(np.array([1.0]))

    def test_raises_neg_bw(self):
        with pytest.raises(ValueError):
            kdist(np.array([1.0, 2.0]), bw=-1)
