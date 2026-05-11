"""Tests for morie.fn.pauli -- Pauli matrices."""

import numpy as np
import pytest

from morie.fn.pauli import pauli


def test_all_returns_six():
    r = pauli("all")
    assert len(r) == 6
    for k in ("sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus", "identity"):
        assert k in r


def test_pauli_algebra():
    r = pauli("all")
    sx, sy, sz = r["sigma_x"], r["sigma_y"], r["sigma_z"]
    np.testing.assert_allclose(sx @ sx, np.eye(2, dtype=complex), atol=1e-14)
    np.testing.assert_allclose(sy @ sy, np.eye(2, dtype=complex), atol=1e-14)
    np.testing.assert_allclose(sz @ sz, np.eye(2, dtype=complex), atol=1e-14)


def test_commutator_xy():
    r = pauli("all")
    comm = r["sigma_x"] @ r["sigma_y"] - r["sigma_y"] @ r["sigma_x"]
    expected = 2j * r["sigma_z"]
    np.testing.assert_allclose(comm, expected, atol=1e-14)


def test_n_dot_sigma_eigenvalues():
    r = pauli("n_dot_sigma", theta=np.pi / 4, phi=0)
    np.testing.assert_allclose(r["eigenvalues"], [-1.0, 1.0], atol=1e-10)


def test_single_operator():
    r = pauli("x")
    assert r["operator"].shape == (2, 2)


def test_unknown_raises():
    with pytest.raises(ValueError):
        pauli("q")
