"""Tests for morie.fn.entqm -- von Neumann entropy."""

import numpy as np
import pytest

from morie.fn.entqm import entqm


def test_returns_dict():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    r = entqm(rho)
    assert isinstance(r, dict)
    for k in ("entropy", "eigenvalues", "purity", "is_pure", "rank"):
        assert k in r


def test_pure_state_zero_entropy():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    r = entqm(rho)
    assert r["entropy"] == pytest.approx(0.0, abs=1e-10)
    assert r["is_pure"] is True
    assert r["purity"] == pytest.approx(1.0, abs=1e-10)


def test_maximally_mixed():
    rho = 0.5 * np.eye(2, dtype=complex)
    r = entqm(rho)
    assert r["entropy"] == pytest.approx(np.log(2), abs=1e-10)
    assert r["is_pure"] is False


def test_bits():
    rho = 0.5 * np.eye(2, dtype=complex)
    r = entqm(rho, base=2.0)
    assert r["entropy"] == pytest.approx(1.0, abs=1e-10)


def test_rank():
    rho = np.diag([0.5, 0.3, 0.2, 0.0]).astype(complex)
    r = entqm(rho)
    assert r["rank"] == 3
