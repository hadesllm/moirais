"""Tests for ltcfl.py - Lattice filter coefficients."""
import numpy as np
from morie.fn.ltcfl import lattice_coefficients_fn, ltcfl


def test_ltcfl_returns_result():
    ar = np.array([0.5, -0.3, 0.1])
    result = lattice_coefficients_fn(ar)
    assert result.name == "lattice_coefficients"
    assert "lattice_coeffs" in result.extra
    assert len(result.extra["lattice_coeffs"]) == 3


def test_ltcfl_alias():
    ar = np.array([0.4, -0.2])
    result = ltcfl(ar)
    assert result.name == "lattice_coefficients"
