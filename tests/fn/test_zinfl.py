"""Tests for morie.fn.zinfl — Zero-inflated Poisson."""

import numpy as np
import pytest

from morie.fn.zinfl import zero_inflated_poisson


def test_zip_detects_inflation():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    inflate = rng.uniform(size=n) < 0.3
    mu = np.exp(1.0 + 0.5 * X[:, 0])
    y = np.where(inflate, 0, rng.poisson(mu))
    res = zero_inflated_poisson(y.astype(float), X)
    assert res.extra["psi_mean"] > 0.1
    assert res.extra["n_zeros"] > 0.25 * n


def test_zip_count_coef_positive():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    inflate = rng.uniform(size=n) < 0.2
    mu = np.exp(0.5 + 1.0 * X[:, 0])
    y = np.where(inflate, 0, rng.poisson(mu))
    res = zero_inflated_poisson(y.astype(float), X)
    assert res.coefficients["count_x0"] > 0


def test_zip_aic_finite():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 1))
    y = rng.poisson(lam=np.exp(0.5 * X[:, 0]))
    y[:30] = 0
    res = zero_inflated_poisson(y.astype(float), X)
    assert np.isfinite(res.extra["aic"])
