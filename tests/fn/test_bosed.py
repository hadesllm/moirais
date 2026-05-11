"""Tests for morie.fn.bosed -- Bose-Einstein distribution."""

import numpy as np
import pytest

from morie.fn.bosed import bosed


def test_returns_dict():
    E = np.linspace(1e-21, 1e-19, 50)
    r = bosed(E, mu=0.0, T=300.0)
    assert isinstance(r, dict)
    for k in ("occupation", "mean_occupation", "total_energy"):
        assert k in r


def test_occupation_positive():
    E = np.linspace(1e-21, 1e-19, 50)
    r = bosed(E, mu=0.0, T=300.0)
    assert np.all(r["occupation"] > 0)


def test_high_energy_low_occupation():
    E = np.array([1e-17])
    r = bosed(E, mu=0.0, T=300.0)
    assert r["occupation"][0] < 0.01


def test_energy_below_mu_raises():
    with pytest.raises(ValueError, match="energies must be > mu"):
        bosed(np.array([0.5e-19]), mu=1e-19, T=300.0)


def test_zero_temp_raises():
    with pytest.raises(ValueError):
        bosed(np.array([1e-19]), mu=0.0, T=0.0)
