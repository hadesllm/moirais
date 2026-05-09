"""Tests for moirais.fn.redsh -- cosmological redshift."""

import numpy as np
import pytest

from moirais.fn.redsh import redsh


def test_returns_dict():
    r = redsh(z=1.0)
    assert isinstance(r, dict)
    for k in ("z", "scale_factor_ratio", "velocity_approx_km_s",
              "velocity_relativistic_km_s"):
        assert k in r


def test_z_from_wavelengths():
    r = redsh(wavelength_obs=700.0, wavelength_emit=500.0)
    assert r["z"] == pytest.approx(0.4, rel=1e-10)


def test_z_from_scale_factor():
    r = redsh(a_emit=0.5)
    assert r["z"] == pytest.approx(1.0, rel=1e-10)


def test_z_zero():
    r = redsh(z=0.0)
    assert r["velocity_approx_km_s"] == pytest.approx(0.0, abs=1e-10)
    assert r["velocity_relativistic_km_s"] == pytest.approx(0.0, abs=1e-10)


def test_relativistic_less_than_c():
    r = redsh(z=10.0)
    assert r["velocity_relativistic_km_s"] < 299792.458


def test_no_input_raises():
    with pytest.raises(ValueError):
        redsh()
