"""Tests for morie.fn.hurds — Hurdle model."""

import numpy as np
import pytest

from morie.fn.hurds import hurdle_model


def test_hurdle_separates_parts():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    prob_pos = 1 / (1 + np.exp(-1.0 - 0.5 * X[:, 0]))
    is_pos = rng.uniform(size=n) < prob_pos
    mu = np.exp(0.5 + 1.0 * X[:, 0])
    y = np.where(is_pos, rng.poisson(mu) + 1, 0).astype(float)
    res = hurdle_model(y, X)
    assert "hurdle_(Intercept)" in res.coefficients
    assert "count_(Intercept)" in res.coefficients
    assert res.extra["n_zeros"] > 50
    assert res.extra["n_positive"] > 100


def test_hurdle_fitted_nonneg():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = rng.poisson(lam=np.exp(0.5 * X[:, 0])).astype(float)
    y[:40] = 0
    res = hurdle_model(y, X)
    assert np.all(res.fitted >= -0.1)


def test_hurdle_ll_finite():
    rng = np.random.default_rng(7)
    n = 300
    X = rng.standard_normal((n, 1))
    y = rng.poisson(lam=2.0, size=n).astype(float)
    y[:60] = 0
    res = hurdle_model(y, X)
    assert np.isfinite(res.extra["log_likelihood"])
