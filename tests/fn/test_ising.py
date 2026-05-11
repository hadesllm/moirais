"""Tests for morie.fn.ising -- 2D Ising model."""

import numpy as np
import pytest

from morie.fn.ising import ising


def test_returns_dict():
    r = ising(T=2.0)
    assert isinstance(r, dict)
    for k in ("T_c", "beta_c", "magnetization", "free_energy_per_site",
              "internal_energy_per_site", "is_ordered"):
        assert k in r


def test_critical_temperature():
    r = ising(T=2.0, J=1.0, kB=1.0)
    Tc_exact = 2.0 / np.log(1 + np.sqrt(2))
    assert r["T_c"] == pytest.approx(Tc_exact, rel=1e-10)


def test_ordered_below_Tc():
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    r = ising(T=0.5 * Tc)
    assert r["is_ordered"] is True
    assert r["magnetization"] > 0


def test_disordered_above_Tc():
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    r = ising(T=1.5 * Tc)
    assert r["is_ordered"] is False
    assert r["magnetization"] == pytest.approx(0.0, abs=1e-10)


def test_zero_temp_raises():
    with pytest.raises(ValueError):
        ising(T=0.0)


def test_negative_J_raises():
    with pytest.raises(ValueError, match="J must be > 0"):
        ising(T=2.0, J=-1.0)
