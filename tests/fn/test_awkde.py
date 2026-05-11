"""Tests for morie.fn.awkde — adaptive bandwidth KDE."""

import numpy as np
import pytest

from morie.fn.awkde import awkde


class TestAwkde:
    def test_integrates_near_one(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 300)
        res = awkde(data, n_grid=512)
        area = np.trapezoid(res["density"], res["x_eval"])
        assert area == pytest.approx(1.0, abs=0.1)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = awkde(data)
        assert np.all(res["density"] >= -1e-12)

    def test_alpha_zero_equals_fixed(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        res = awkde(data, alpha=0.0, bw=0.5, n_grid=64)
        assert res["alpha"] == 0.0

    def test_raises_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            awkde(np.array([1.0, 2.0, 3.0]), alpha=1.5)

    def test_raises_small(self):
        with pytest.raises(ValueError):
            awkde(np.array([1.0]))
