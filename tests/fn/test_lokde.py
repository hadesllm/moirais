"""Tests for morie.fn.lokde — local polynomial density estimator."""

import numpy as np
import pytest

from morie.fn.lokde import lokde


class TestLokde:
    def test_integrates_near_one(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = lokde(data, n_grid=256)
        area = np.trapezoid(res["density"], res["x_eval"])
        assert area == pytest.approx(1.0, abs=0.15)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        res = lokde(data, n_grid=128)
        assert np.all(res["density"] >= -1e-10)

    def test_degree_param(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res = lokde(data, degree=1, n_grid=64)
        assert res["degree"] == 1

    def test_raises_small(self):
        with pytest.raises(ValueError):
            lokde(np.array([1.0]))
