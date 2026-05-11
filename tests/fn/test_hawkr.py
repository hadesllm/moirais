"""Tests for morie.fn.hawkr -- Hawking radiation temperature."""

import numpy as np
import pytest

from morie.fn.hawkr import hawkr


def test_returns_dict():
    r = hawkr(M=1.989e30)
    assert isinstance(r, dict)
    for k in ("temperature_K", "r_schwarzschild", "luminosity_W",
              "evaporation_time_s", "peak_wavelength_m"):
        assert k in r


def test_solar_mass_cold():
    r = hawkr(M=1.989e30)
    assert r["temperature_K"] < 1e-6


def test_small_mass_hot():
    r = hawkr(M=1e10)
    assert r["temperature_K"] > 1e10


def test_temperature_formula():
    M = 1e20
    hbar = 1.0545718e-34
    c = 299792458.0
    G = 6.67430e-11
    kB = 1.380649e-23
    expected = hbar * c ** 3 / (8 * np.pi * G * M * kB)
    r = hawkr(M=M)
    assert r["temperature_K"] == pytest.approx(expected, rel=1e-6)


def test_evaporation_time_positive():
    r = hawkr(M=1e15)
    assert r["evaporation_time_s"] > 0


def test_negative_mass_raises():
    with pytest.raises(ValueError):
        hawkr(M=-1.0)
