"""Tests for morie.fn.densm -- density matrix operations."""

import numpy as np
import pytest

from morie.fn.densm import densm


def test_construct():
    psi = np.array([1, 0], dtype=complex)
    r = densm(state=psi, operation="construct")
    assert "rho_out" in r
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    np.testing.assert_allclose(r["rho_out"], expected, atol=1e-14)


def test_construct_normalizes():
    psi = np.array([3, 4], dtype=complex)
    r = densm(state=psi, operation="construct")
    assert r["trace"] == pytest.approx(1.0, abs=1e-10)


def test_partial_trace_bell_state():
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, np.conj(psi))
    r = densm(rho=rho, operation="partial_trace",
              subsystem_dims=(2, 2), trace_out=1)
    expected = 0.5 * np.eye(2, dtype=complex)
    np.testing.assert_allclose(r["rho_out"], expected, atol=1e-10)


def test_purity():
    rho = 0.5 * np.eye(2, dtype=complex)
    r = densm(rho=rho, operation="purity")
    assert r["purity"] == pytest.approx(0.5, abs=1e-10)
    assert r["is_pure"] is False


def test_zero_state_raises():
    with pytest.raises(ValueError, match="Zero"):
        densm(state=np.zeros(2), operation="construct")
