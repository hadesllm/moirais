"""Tests for moirais.fn.wdist — Wasserstein distance."""

import numpy as np
import pytest

from moirais.fn.wdist import wdist


class TestWdist:
    def test_identical_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        result = wdist(x, x)
        assert result["distance"] == pytest.approx(0.0, abs=1e-10)

    def test_shift(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        result = wdist(x, y, p=1)
        assert result["distance"] == pytest.approx(1.0, abs=1e-10)

    def test_w2(self):
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 1.0])
        result = wdist(x, y, p=2)
        assert result["distance"] == pytest.approx(1.0, abs=1e-10)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(1, 1, 100)
        assert wdist(x, y)["distance"] >= 0

    def test_different_lengths(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.5, 1.0, 1.5])
        result = wdist(x, y)
        assert result["distance"] >= 0

    def test_empty_error(self):
        with pytest.raises(ValueError):
            wdist(np.array([]), np.array([1.0]))

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            wdist(np.array([1.0]), np.array([2.0]), p=0)
