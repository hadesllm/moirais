"""Tests for moirais.fn.kbias — bias-reduced KDE."""

import numpy as np
import pytest

from moirais.fn.kbias import kbias


class TestKbias:
    def test_integrates_near_one(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 300)
        res = kbias(data)
        area = np.trapezoid(res["density"], res["x_eval"])
        assert area == pytest.approx(1.0, abs=0.1)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kbias(data)
        assert np.all(res["density"] >= 0)

    def test_peak_near_mode(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5, 0.5, 500)
        res = kbias(data, n_grid=256)
        peak_x = res["x_eval"][np.argmax(res["density"])]
        assert peak_x == pytest.approx(5.0, abs=0.3)

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbias(np.array([1.0]))
