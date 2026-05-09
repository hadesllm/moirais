"""Tests for moirais.fn.kbeta — beta kernel density."""

import numpy as np
import pytest

from moirais.fn.kbeta import kbeta


class TestKbeta:
    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        data = rng.beta(2, 5, 200)
        res = kbeta(data)
        assert np.all(res["density"] >= -1e-10)

    def test_integrates_near_one(self):
        rng = np.random.default_rng(42)
        data = rng.beta(2, 5, 300)
        res = kbeta(data, n_grid=512)
        area = np.trapezoid(res["density"], res["x_eval"])
        assert area == pytest.approx(1.0, abs=0.15)

    def test_custom_bounds(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(2, 8, 100)
        res = kbeta(data, lower=2.0, upper=8.0, n_grid=128)
        assert len(res["density"]) == 128

    def test_raises_out_of_bounds(self):
        with pytest.raises(ValueError, match="within"):
            kbeta(np.array([0.5, 1.5]), lower=0.0, upper=1.0)

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbeta(np.array([0.5]))

    def test_raises_bad_bounds(self):
        with pytest.raises(ValueError):
            kbeta(np.array([0.5, 0.6]), lower=1.0, upper=0.0)
