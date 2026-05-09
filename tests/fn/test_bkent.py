"""Tests for moirais.fn.bkent -- Bekenstein-Hawking entropy."""

import numpy as np
import pytest

from moirais.fn.bkent import bkent


def test_returns_dict():
    r = bkent(M=1.989e30)
    assert isinstance(r, dict)
    for k in ("entropy_J_per_K", "entropy_bits", "horizon_area_m2",
              "r_schwarzschild", "temperature_K"):
        assert k in r


def test_entropy_positive():
    r = bkent(M=1.989e30)
    assert r["entropy_J_per_K"] > 0
    assert r["entropy_bits"] > 0


def test_area_from_mass():
    M = 1e30
    G = 6.67430e-11
    c = 299792458.0
    rs = 2 * G * M / c ** 2
    expected_A = 4 * np.pi * rs ** 2
    r = bkent(M=M)
    assert r["horizon_area_m2"] == pytest.approx(expected_A, rel=1e-6)


def test_from_area():
    A = 1e6
    r = bkent(A=A)
    assert r["horizon_area_m2"] == pytest.approx(A, rel=1e-10)
    assert r["entropy_J_per_K"] > 0


def test_temperature_matches_hawking():
    M = 1e20
    r = bkent(M=M)
    hbar = 1.0545718e-34
    c = 299792458.0
    G = 6.67430e-11
    kB = 1.380649e-23
    expected_T = hbar * c ** 3 / (8 * np.pi * G * M * kB)
    assert r["temperature_K"] == pytest.approx(expected_T, rel=1e-6)


def test_no_input_raises():
    with pytest.raises(ValueError):
        bkent()
