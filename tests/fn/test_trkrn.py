"""Tests for moirais.fn.trkrn — triweight kernel."""

import numpy as np
import pytest

from moirais.fn.trkrn import trkrn


class TestTrkrn:
    def test_at_zero(self):
        assert trkrn(0.0) == pytest.approx(35.0 / 32.0)

    def test_at_boundary(self):
        assert trkrn(1.0) == pytest.approx(0.0)

    def test_outside_support(self):
        assert trkrn(2.0) == 0.0

    def test_symmetry(self):
        assert trkrn(0.4) == pytest.approx(trkrn(-0.4))

    def test_integrates_to_one(self):
        u = np.linspace(-1, 1, 10000)
        area = np.trapezoid(trkrn(u), u)
        assert area == pytest.approx(1.0, abs=1e-4)

    def test_array(self):
        result = trkrn(np.array([-0.5, 0.0, 0.5]))
        assert isinstance(result, np.ndarray)
