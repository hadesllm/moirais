"""Tests for morie.fn.epkrn — Epanechnikov kernel."""

import numpy as np
import pytest

from morie.fn.epkrn import epkrn


class TestEpkrn:
    def test_at_zero(self):
        assert epkrn(0.0) == pytest.approx(0.75)

    def test_at_boundary(self):
        assert epkrn(1.0) == pytest.approx(0.0)
        assert epkrn(-1.0) == pytest.approx(0.0)

    def test_outside_support(self):
        assert epkrn(1.5) == 0.0
        assert epkrn(-2.0) == 0.0

    def test_symmetry(self):
        assert epkrn(0.5) == pytest.approx(epkrn(-0.5))

    def test_integrates_to_one(self):
        u = np.linspace(-1, 1, 10000)
        area = np.trapezoid(epkrn(u), u)
        assert area == pytest.approx(1.0, abs=1e-4)

    def test_array_input(self):
        result = epkrn(np.array([-0.5, 0.0, 0.5]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
