"""Tests for moirais.fn.cosmx -- comoving distance."""

import numpy as np
import pytest

from moirais.fn.cosmx import cosmx


def test_returns_dict():
    r = cosmx(z=1.0)
    assert isinstance(r, dict)
    for k in ("comoving_distance_Mpc", "luminosity_distance_Mpc",
              "angular_diameter_distance_Mpc", "lookback_time_Gyr"):
        assert k in r


def test_z_zero():
    r = cosmx(z=0.0)
    assert r["comoving_distance_Mpc"] == pytest.approx(0.0, abs=1e-10)
    assert r["lookback_time_Gyr"] == pytest.approx(0.0, abs=1e-10)


def test_distances_positive():
    r = cosmx(z=1.0)
    assert r["comoving_distance_Mpc"] > 0
    assert r["luminosity_distance_Mpc"] > r["comoving_distance_Mpc"]
    assert r["angular_diameter_distance_Mpc"] < r["comoving_distance_Mpc"]


def test_dl_relation():
    z = 2.0
    r = cosmx(z=z)
    ratio = r["luminosity_distance_Mpc"] / r["comoving_distance_Mpc"]
    assert ratio == pytest.approx(1 + z, rel=1e-6)


def test_negative_z_raises():
    with pytest.raises(ValueError):
        cosmx(z=-0.5)
