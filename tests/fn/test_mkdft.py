"""Tests for moirais.fn.mkdft — multivariate product kernel density."""

import numpy as np
import pytest

from moirais.fn.mkdft import mkdft


class TestMkdft:
    def test_2d_nonnegative(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 2))
        res = mkdft(data, n_grid=20)
        assert np.all(res["density"] >= 0)

    def test_1d_fallback(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)
        res = mkdft(data, n_grid=30)
        assert res["bw"].shape == (1,)

    def test_custom_bw_scalar(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (50, 2))
        res = mkdft(data, bw=0.5, n_grid=10)
        np.testing.assert_array_equal(res["bw"], [0.5, 0.5])

    def test_custom_eval_points(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (50, 2))
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        res = mkdft(data, x_eval=pts)
        assert len(res["density"]) == 2

    def test_raises_small(self):
        with pytest.raises(ValueError):
            mkdft(np.array([[1.0, 2.0]]))

    def test_raises_bw_mismatch(self):
        with pytest.raises(ValueError):
            mkdft(np.ones((10, 3)), bw=np.array([1.0, 2.0]))
