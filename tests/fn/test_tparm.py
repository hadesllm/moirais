"""Tests for morie.fn.tparm — torus parametric coordinates."""

import numpy as np
import pytest

from morie.fn.tparm import torus_parametric


class TestTorusParametric:
    def test_origin(self):
        r = torus_parametric(R=3, r=1, u=0, v=0)
        assert r.extra["x"] == pytest.approx(4.0)
        assert r.extra["y"] == pytest.approx(0.0, abs=1e-15)
        assert r.extra["z"] == pytest.approx(0.0, abs=1e-15)

    def test_array_input(self):
        u = np.array([0, np.pi])
        v = np.array([0, 0])
        r = torus_parametric(R=3, r=1, u=u, v=v)
        assert len(r.extra["x"]) == 2

    def test_name(self):
        assert torus_parametric().name == "torus_parametric"

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_parametric(R=-1, r=1)
