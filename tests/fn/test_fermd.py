"""Tests for moirais.fn.fermd -- Fermi-Dirac distribution."""

import numpy as np
import pytest

from moirais.fn.fermd import fermd


def test_returns_dict():
    E = np.linspace(0, 2e-19, 50)
    r = fermd(E, mu=1e-19, T=300.0)
    assert isinstance(r, dict)
    for k in ("occupation", "mean_occupation", "total_energy"):
        assert k in r


def test_at_fermi_energy():
    mu = 1e-19
    r = fermd(np.array([mu]), mu=mu, T=300.0)
    assert r["occupation"][0] == pytest.approx(0.5, abs=1e-10)


def test_occupation_between_0_and_1():
    E = np.linspace(0, 5e-19, 100)
    r = fermd(E, mu=2e-19, T=300.0)
    assert np.all(r["occupation"] >= 0)
    assert np.all(r["occupation"] <= 1)


def test_step_function_at_low_T():
    E = np.array([0.5e-19, 1.5e-19])
    mu = 1e-19
    r = fermd(E, mu=mu, T=0.01)
    assert r["occupation"][0] > 0.999
    assert r["occupation"][1] < 0.001


def test_zero_temp_raises():
    with pytest.raises(ValueError):
        fermd(np.array([1e-19]), mu=0.5e-19, T=0.0)
