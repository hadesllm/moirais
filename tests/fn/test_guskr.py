"""Tests for moirais.fn.guskr — Gaussian kernel."""

import numpy as np
import pytest

from moirais.fn.guskr import guskr


class TestGuskr:
    def test_at_zero(self):
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert guskr(0.0) == pytest.approx(expected, rel=1e-10)

    def test_symmetry(self):
        assert guskr(1.0) == pytest.approx(guskr(-1.0))

    def test_integrates_to_one(self):
        u = np.linspace(-10, 10, 100000)
        area = np.trapezoid(guskr(u), u)
        assert area == pytest.approx(1.0, abs=1e-4)

    def test_tails_near_zero(self):
        assert guskr(5.0) < 1e-5

    def test_array_input(self):
        result = guskr(np.array([-1.0, 0.0, 1.0]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
