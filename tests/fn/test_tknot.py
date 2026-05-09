"""Tests for moirais.fn.tknot — torus knot."""

import numpy as np
import pytest

from moirais.fn.tknot import torus_knot


class TestTorusKnot:
    def test_trefoil(self):
        r = torus_knot(p=2, q=3, n_points=100)
        assert len(r.extra["x"]) == 100

    def test_name(self):
        assert torus_knot().name == "torus_knot"

    def test_on_torus(self):
        r = torus_knot(p=2, q=3, R=3, r=1, n_points=50)
        rho = np.sqrt(r.extra["x"]**2 + r.extra["y"]**2)
        assert np.all(rho >= 2) and np.all(rho <= 4)

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_knot(p=0, q=3)
