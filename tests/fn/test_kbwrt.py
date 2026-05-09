"""Tests for moirais.fn.kbwrt — Silverman's rule-of-thumb bandwidth."""

import numpy as np
import pytest

from moirais.fn.kbwrt import kbwrt


class TestKbwrt:
    def test_gaussian_normal_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        res = kbwrt(data)
        assert res["bw"] > 0
        assert res["kernel"] == "gaussian"

    def test_silverman_formula(self):
        data = np.arange(1, 101, dtype=float)
        res = kbwrt(data)
        sigma = np.std(data, ddof=1)
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        s = min(sigma, iqr / 1.349)
        expected = 0.9 * s * 100 ** (-0.2)
        assert res["bw"] == pytest.approx(expected, rel=1e-6)

    def test_epanechnikov_wider(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        gauss = kbwrt(data, kernel="gaussian")
        epan = kbwrt(data, kernel="epanechnikov")
        assert epan["bw"] > gauss["bw"]

    def test_raises_unknown_kernel(self):
        with pytest.raises(ValueError, match="Unknown"):
            kbwrt(np.array([1.0, 2.0, 3.0]), kernel="bogus")

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbwrt(np.array([1.0]))
