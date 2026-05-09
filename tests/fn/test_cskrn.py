"""Tests for moirais.fn.cskrn — cosine kernel."""

import numpy as np
import pytest

from moirais.fn.cskrn import cskrn


class TestCskrn:
    def test_at_zero(self):
        assert cskrn(0.0) == pytest.approx(np.pi / 4.0)

    def test_at_boundary(self):
        assert cskrn(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_outside_support(self):
        assert cskrn(1.5) == 0.0

    def test_symmetry(self):
        assert cskrn(0.3) == pytest.approx(cskrn(-0.3))

    def test_integrates_to_one(self):
        u = np.linspace(-1, 1, 10000)
        area = np.trapezoid(cskrn(u), u)
        assert area == pytest.approx(1.0, abs=1e-3)

    def test_array(self):
        result = cskrn(np.array([-0.5, 0.0, 0.5]))
        assert isinstance(result, np.ndarray)
