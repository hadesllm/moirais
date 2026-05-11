"""Tests for morie.fn.irfun — Impulse response function."""

import numpy as np
import pytest

from morie.fn.irfun import impulse_response, irfun


def _var1_params(m: int = 2, diag_coef: float = 0.5) -> tuple:
    """Simple stable VAR(1) with diagonal AR matrix and identity sigma."""
    A = np.diag([diag_coef] * m)
    coef = np.hstack([np.zeros((m, 1)), A])   # (m, 1+m*1)
    sigma_u = np.eye(m)
    return coef, sigma_u


def test_returns_descriptive_result():
    """Return type has the DescriptiveResult interface."""
    coef, sigma_u = _var1_params(m=2)
    r = impulse_response(coef, sigma_u, horizon=5, shock_var=0)
    assert hasattr(r, "value")
    assert hasattr(r, "extra")
    assert "irf" in r.extra


def test_irf_shape():
    """IRF array has shape (horizon+1, m)."""
    m = 3
    coef, sigma_u = _var1_params(m=m)
    r = impulse_response(coef, sigma_u, horizon=10, shock_var=0)
    assert r.extra["irf"].shape == (11, m)


def test_irf_h0_equals_cholesky_col():
    """At h=0, IRF[0] = Phi[0] @ P[:, shock] = I @ P[:, 0] = P[:, 0]."""
    coef, sigma_u = _var1_params(m=2)
    P = np.linalg.cholesky(sigma_u)
    r = impulse_response(coef, sigma_u, horizon=5, shock_var=0)
    np.testing.assert_allclose(r.extra["irf"][0], P[:, 0], atol=1e-12)


def test_stable_var_irf_decays():
    """For a stable VAR (eigenvalues < 1), IRF should decay toward 0."""
    coef, sigma_u = _var1_params(m=2, diag_coef=0.5)
    r = impulse_response(coef, sigma_u, horizon=30, shock_var=0)
    irf = r.extra["irf"]
    # Response at h=30 must be smaller in absolute value than at h=1.
    assert np.abs(irf[30]).max() < np.abs(irf[1]).max()


def test_irf_identity_sigma():
    """With sigma_u=I, Cholesky factor P=I, so IRF[0] = e_shock."""
    m = 3
    shock = 1
    coef, sigma_u = _var1_params(m=m)
    r = impulse_response(coef, sigma_u, horizon=5, shock_var=shock)
    irf0 = r.extra["irf"][0]
    e_shock = np.eye(m)[:, shock]
    np.testing.assert_allclose(irf0, e_shock, atol=1e-12)


def test_phi_shape():
    """Phi matrices have shape (horizon+1, m, m)."""
    coef, sigma_u = _var1_params(m=2)
    r = impulse_response(coef, sigma_u, horizon=8)
    assert r.extra["Phi"].shape == (9, 2, 2)


def test_invalid_shock_var_raises():
    coef, sigma_u = _var1_params(m=2)
    with pytest.raises(ValueError):
        impulse_response(coef, sigma_u, shock_var=5)


def test_alias():
    assert irfun is impulse_response
