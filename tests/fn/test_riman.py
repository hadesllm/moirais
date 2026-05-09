"""Tests for moirais.fn.riman -- Riemann curvature tensor."""

import numpy as np
import pytest

from moirais.fn.riman import riman


def test_returns_dict():
    G = np.zeros((4, 4, 4))
    r = riman(G)
    assert isinstance(r, dict)
    assert "riemann" in r


def test_zero_christoffel_zero_riemann():
    G = np.zeros((4, 4, 4))
    r = riman(G)
    np.testing.assert_allclose(r["riemann"], 0.0, atol=1e-14)


def test_antisymmetry_last_two():
    G = np.random.default_rng(42).standard_normal((4, 4, 4)) * 0.01
    dG = np.random.default_rng(43).standard_normal((4, 4, 4, 4)) * 0.01
    r = riman(G, christoffel_derivs=dG)
    R = r["riemann"]
    for rho in range(4):
        for sig in range(4):
            for mu in range(4):
                for nu in range(4):
                    assert R[rho, sig, mu, nu] == pytest.approx(
                        -R[rho, sig, nu, mu], abs=1e-12
                    )


def test_shape():
    G = np.zeros((3, 3, 3))
    r = riman(G)
    assert r["riemann"].shape == (3, 3, 3, 3)
