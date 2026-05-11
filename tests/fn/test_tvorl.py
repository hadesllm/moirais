"""Tests for morie.fn.tvorl — Voronoi on flat torus."""

import numpy as np
import pytest

from morie.fn.tvorl import torus_voronoi


class TestTorusVoronoi:
    def test_default(self):
        r = torus_voronoi()
        assert r.extra["assignment"].shape == (20, 20)
        assert r.extra["n_seeds"] == 10

    def test_custom_seeds(self):
        pts = np.array([[0.2, 0.3], [0.7, 0.8]])
        r = torus_voronoi(points=pts, n=10)
        assert r.extra["n_seeds"] == 2

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_voronoi(a=0)
