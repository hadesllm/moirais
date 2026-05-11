"""Tests for morie.fn.torus — torus surface area and volume."""

import numpy as np
import pytest

from morie.fn.torus import torus_surface


class TestTorusSurface:
    def test_standard_torus(self):
        r = torus_surface(R=3.0, r=1.0)
        assert r.extra["surface_area"] == pytest.approx(4 * np.pi**2 * 3 * 1, rel=1e-10)

    def test_volume(self):
        r = torus_surface(R=3.0, r=1.0)
        assert r.extra["volume"] == pytest.approx(2 * np.pi**2 * 3 * 1, rel=1e-10)

    def test_name(self):
        assert torus_surface().name == "torus_surface"

    def test_invalid_radii(self):
        with pytest.raises(ValueError):
            torus_surface(R=0, r=1)
        with pytest.raises(ValueError):
            torus_surface(R=3, r=-1)

    def test_r_greater_than_R(self):
        with pytest.raises(ValueError):
            torus_surface(R=1, r=2)
