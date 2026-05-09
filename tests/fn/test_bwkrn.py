"""Tests for moirais.fn.bwkrn — biweight kernel."""

import numpy as np
import pytest

from moirais.fn.bwkrn import bwkrn


class TestBwkrn:
    def test_at_zero(self):
        assert bwkrn(0.0) == pytest.approx(15.0 / 16.0)

    def test_at_boundary(self):
        assert bwkrn(1.0) == pytest.approx(0.0)

    def test_outside_support(self):
        assert bwkrn(1.5) == 0.0

    def test_symmetry(self):
        assert bwkrn(0.3) == pytest.approx(bwkrn(-0.3))

    def test_integrates_to_one(self):
        u = np.linspace(-1, 1, 10000)
        area = np.trapezoid(bwkrn(u), u)
        assert area == pytest.approx(1.0, abs=1e-4)

    def test_array_input(self):
        result = bwkrn(np.array([-0.5, 0.0, 0.5]))
        assert isinstance(result, np.ndarray)
