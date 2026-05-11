"""Tests for morie.fn.einsf -- Einstein field equation tensor."""

import numpy as np
import pytest

from morie.fn.einsf import einsf


def test_returns_dict():
    R_mn = np.zeros((4, 4))
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = einsf(R_mn, 0.0, g)
    assert isinstance(r, dict)
    for k in ("einstein_tensor", "residual", "kappa"):
        assert k in r


def test_flat_space_zero_einstein():
    R_mn = np.zeros((4, 4))
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = einsf(R_mn, 0.0, g)
    np.testing.assert_allclose(r["einstein_tensor"], 0.0, atol=1e-14)


def test_kappa_value():
    R_mn = np.zeros((4, 4))
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    r = einsf(R_mn, 0.0, g)
    G_val = 6.67430e-11
    c_val = 299792458.0
    expected = 8.0 * np.pi * G_val / c_val ** 4
    assert r["kappa"] == pytest.approx(expected, rel=1e-10)


def test_residual_with_stress_energy():
    R_mn = np.eye(4) * 0.1
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    T = np.eye(4) * 1e10
    r = einsf(R_mn, 0.5, g, stress_energy=T)
    assert r["residual"] is not None
    assert r["residual"].shape == (4, 4)


def test_no_stress_energy_no_residual():
    r = einsf(np.zeros((4, 4)), 0.0, np.eye(4))
    assert r["residual"] is None
