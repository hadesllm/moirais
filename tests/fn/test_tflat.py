"""Tests for morie.fn.tflat — flat torus."""

import pytest

from morie.fn.tflat import flat_torus


class TestFlatTorus:
    def test_unit_square(self):
        r = flat_torus(a=1, b=1, n=10)
        assert r.value == 1.0
        assert r.extra["curvature"] == 0.0

    def test_grid_shape(self):
        r = flat_torus(a=2, b=3, n=5)
        assert r.extra["grid_x"].shape == (5, 5)

    def test_invalid(self):
        with pytest.raises(ValueError):
            flat_torus(a=0, b=1)
        with pytest.raises(ValueError):
            flat_torus(a=1, b=1, n=1)
