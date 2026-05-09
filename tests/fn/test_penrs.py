"""Tests for moirais.fn.penrs -- Penrose diagram coordinates."""

import numpy as np
import pytest

from moirais.fn.penrs import penrs


def test_returns_dict():
    r = penrs(r=np.array([1.0, 2.0]), t=np.array([0.0, 0.0]),
              geometry="minkowski")
    assert isinstance(r, dict)
    for k in ("U", "V", "T_penrose", "R_penrose"):
        assert k in r


def test_minkowski_origin():
    r = penrs(r=np.array([0.0]), t=np.array([0.0]), geometry="minkowski")
    assert r["U"][0] == pytest.approx(0.0, abs=1e-10)
    assert r["V"][0] == pytest.approx(0.0, abs=1e-10)


def test_minkowski_compactified_range():
    r_vals = np.linspace(0, 1000, 100)
    t_vals = np.zeros(100)
    r = penrs(r=r_vals, t=t_vals, geometry="minkowski")
    assert np.all(r["U"] >= -np.pi / 2 - 0.01)
    assert np.all(r["V"] <= np.pi / 2 + 0.01)


def test_schwarzschild():
    r_vals = np.array([10.0, 20.0, 50.0])
    t_vals = np.array([0.0, 0.0, 0.0])
    r = penrs(r=r_vals, t=t_vals, M=1.0, geometry="schwarzschild",
              G=1.0, c=1.0)
    assert r["R_penrose"].shape == (3,)


def test_inside_horizon_raises():
    with pytest.raises(ValueError, match="Schwarzschild radius"):
        penrs(r=np.array([0.5]), t=np.array([0.0]),
              M=1.0, geometry="schwarzschild", G=1.0, c=1.0)


def test_unknown_geometry_raises():
    with pytest.raises(ValueError, match="geometry"):
        penrs(r=np.array([1.0]), t=np.array([0.0]), geometry="kerr")
