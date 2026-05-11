"""Tests for morie.fn.tdist — geodesic distance on torus."""

import numpy as np
import pytest

from morie.fn.tdist import torus_distance


class TestTorusDistance:
    def test_same_point(self):
        r = torus_distance(p1=(0, 0), p2=(0, 0), R=3, r=1)
        assert r.value == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        r = torus_distance(p1=(0, 0), p2=(np.pi, 0), R=3, r=1)
        assert r.value > 0

    def test_chord_leq_geodesic(self):
        r = torus_distance(p1=(0, 0), p2=(1.0, 0.5), R=3, r=1)
        assert r.extra["chord"] <= r.extra["geodesic"] + 1e-6

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_distance(R=0, r=1)
