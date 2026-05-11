"""Tests for morie.fn.plank -- Planck distribution."""

import numpy as np
import pytest

from morie.fn.plank import plank


def test_returns_dict():
    nu = np.linspace(1e10, 1e14, 100)
    r = plank(nu, T=5778.0)
    assert isinstance(r, dict)
    for k in ("spectral_radiance", "peak_frequency", "total_power", "wien_peak"):
        assert k in r


def test_spectral_radiance_positive():
    nu = np.linspace(1e10, 1e14, 100)
    r = plank(nu, T=5778.0)
    assert np.all(r["spectral_radiance"] >= 0)


def test_stefan_boltzmann():
    nu = np.linspace(1e10, 1e14, 100)
    r = plank(nu, T=5778.0)
    sigma = 5.670374419e-8
    expected = sigma * 5778.0 ** 4
    assert r["total_power"] == pytest.approx(expected, rel=1e-3)


def test_wien_law():
    r = plank(np.array([1e12]), T=3000.0)
    kB = 1.380649e-23
    h = 6.62607015e-34
    expected_peak = 2.8214393 * kB * 3000.0 / h
    assert r["peak_frequency"] == pytest.approx(expected_peak, rel=1e-6)


def test_zero_temp_raises():
    with pytest.raises(ValueError):
        plank(np.array([1e12]), T=0.0)
