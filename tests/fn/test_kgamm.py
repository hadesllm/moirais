"""Tests for morie.fn.kgamm — gamma kernel density."""

import numpy as np
import pytest

from morie.fn.kgamm import kgamm


class TestKgamm:
    def test_nonnegative_density(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(2.0, 200)
        res = kgamm(data)
        assert np.all(res["density"] >= 0)

    def test_peak_location(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, 500)
        res = kgamm(data, n_grid=256)
        peak_x = res["x_eval"][np.argmax(res["density"])]
        assert peak_x < 2.0

    def test_raises_nonpositive_data(self):
        with pytest.raises(ValueError, match="strictly positive"):
            kgamm(np.array([-1.0, 2.0, 3.0]))

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kgamm(np.array([1.0]))

    def test_custom_bw(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = kgamm(data, bw=0.5)
        assert res["bw"] == 0.5
