"""Tests for morie.fn.schwz -- Schwarzschild metric."""

import numpy as np
import pytest

from morie.fn.schwz import schwz


def test_returns_dict():
    r = schwz(r=1e7, M=1.989e30)
    assert isinstance(r, dict)
    for k in ("metric", "r_schwarzschild", "f"):
        assert k in r


def test_metric_shape():
    r = schwz(r=1e7, M=1.989e30)
    assert r["metric"].shape == (4, 4)


def test_solar_schwarzschild_radius():
    r = schwz(r=1e7, M=1.989e30)
    rs_expected = 2 * 6.67430e-11 * 1.989e30 / 299792458.0 ** 2
    assert r["r_schwarzschild"] == pytest.approx(rs_expected, rel=1e-6)


def test_flat_at_infinity():
    r = schwz(r=1e20, M=1.989e30)
    assert r["f"] == pytest.approx(1.0, abs=1e-6)


def test_metric_diagonal():
    result = schwz(r=1e7, M=1.989e30)
    g = result["metric"]
    assert g[0, 1] == 0.0
    assert g[0, 0] < 0
    assert g[1, 1] > 0


def test_inside_horizon_raises():
    with pytest.raises(ValueError, match="Schwarzschild radius"):
        schwz(r=1.0, M=1.989e30)


def test_negative_mass_raises():
    with pytest.raises(ValueError, match="positive"):
        schwz(r=1e7, M=-1.0)
