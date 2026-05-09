"""Tests for moirais.fn.tlatt — torus lattice."""

import numpy as np
import pytest

from moirais.fn.tlatt import torus_lattice


class TestTorusLattice:
    def test_square_lattice(self):
        r = torus_lattice(a=1, b=1, angle=90, n=2)
        assert r.extra["area"] == pytest.approx(1.0, abs=1e-10)

    def test_n_points(self):
        r = torus_lattice(n=3)
        assert r.extra["n_points"] == 49

    def test_hexagonal(self):
        r = torus_lattice(a=1, b=1, angle=60, n=1)
        assert r.extra["area"] == pytest.approx(np.sin(np.radians(60)), rel=1e-10)

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_lattice(a=0)
