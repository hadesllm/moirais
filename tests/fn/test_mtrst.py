"""Tests for moirais.fn.mtrst -- Martingale residuals."""

import numpy as np
import pytest

from moirais.fn.mtrst import mtrst


@pytest.fixture()
def cox_data():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    beta_true = np.array([0.5, -0.3])
    lp = X @ beta_true
    t = rng.exponential(np.exp(-lp))
    c = rng.exponential(3, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, X, beta_true


def test_returns_dict(cox_data):
    time, event, X, beta = cox_data
    r = mtrst(time, event, X, beta)
    assert isinstance(r, dict)
    assert "residuals" in r and "mean_residual" in r


def test_residuals_shape(cox_data):
    time, event, X, beta = cox_data
    r = mtrst(time, event, X, beta)
    assert r["residuals"].shape == (len(time),)


def test_residuals_bounded(cox_data):
    time, event, X, beta = cox_data
    r = mtrst(time, event, X, beta)
    assert np.all(r["residuals"] <= 1.0 + 1e-10)


def test_cheatsheet():
    from moirais.fn.mtrst import cheatsheet
    assert "martingale" in cheatsheet().lower()
