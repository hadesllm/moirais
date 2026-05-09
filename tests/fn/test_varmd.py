"""Tests for moirais.fn.varmd — VAR(p) model fitting."""

import numpy as np
import pytest

from moirais.fn.varmd import var_model, varmd


def test_returns_descriptive_result():
    """Return type has DescriptiveResult interface."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((80, 2))
    r = var_model(Y, p=1)
    assert hasattr(r, "value")
    assert hasattr(r, "extra")
    assert isinstance(r.extra, dict)


def test_coef_shape_bivariate_lag1():
    """VAR(1) bivariate: coefficient matrix shape (m, 1+m*p)."""
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((100, 2))
    r = var_model(Y, p=1)
    m = 2
    assert r.extra["coef"].shape == (m, 1 + m * 1)
    assert r.extra["sigma_u"].shape == (m, m)


def test_coef_shape_trivariate_lag2():
    """VAR(2) trivariate: correct dimensions."""
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((150, 3))
    r = var_model(Y, p=2)
    m, p = 3, 2
    assert r.extra["coef"].shape == (m, 1 + m * p)
    assert r.extra["n"] == 150
    assert r.extra["p"] == 2
    assert r.extra["m"] == 3


def test_univariate_input():
    """1-D input treated as m=1."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal(60)
    r = var_model(y, p=1)
    assert r.extra["m"] == 1
    assert r.extra["coef"].shape == (1, 2)


def test_residuals_near_zero_mean():
    """Residuals from VAR(1) on white noise should have near-zero mean."""
    rng = np.random.default_rng(4)
    Y = rng.standard_normal((200, 2))
    r = var_model(Y, p=1)
    resid_mean = np.abs(r.extra["residuals"].mean(axis=0))
    assert np.all(resid_mean < 0.5)


def test_sigma_u_positive_semidefinite():
    """sigma_u must be positive semidefinite (all eigenvalues >= 0)."""
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((120, 3))
    r = var_model(Y, p=1)
    eigvals = np.linalg.eigvalsh(r.extra["sigma_u"])
    assert np.all(eigvals >= -1e-10)


def test_aic_bic_finite():
    """AIC and BIC should be finite."""
    rng = np.random.default_rng(6)
    Y = rng.standard_normal((100, 2))
    r = var_model(Y, p=1)
    assert np.isfinite(r.extra["aic"])
    assert np.isfinite(r.extra["bic"])


def test_p_too_large_raises():
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((20, 2))
    with pytest.raises(ValueError):
        var_model(Y, p=15)


def test_alias():
    assert varmd is var_model
