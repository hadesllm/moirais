"""Tests for moirais.fn.tcrv — torus curvature."""

import numpy as np
import pytest

from moirais.fn.tcrv import torus_curvature


class TestTorusCurvature:
    def test_outer_equator(self):
        r = torus_curvature(R=3, r=1, v=0)
        assert r.extra["gaussian_K"] == pytest.approx(1.0 / (1 * 4), rel=1e-10)

    def test_inner_equator(self):
        r = torus_curvature(R=3, r=1, v=np.pi)
        assert r.extra["gaussian_K"] == pytest.approx(-1.0 / (1 * 2), rel=1e-10)

    def test_array(self):
        r = torus_curvature(R=3, r=1, v=np.array([0, np.pi / 2, np.pi]))
        assert len(r.extra["gaussian_K"]) == 3

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_curvature(R=0, r=1)
