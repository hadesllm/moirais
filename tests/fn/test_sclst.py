"""Tests for moirais.fn.sclst -- Semiparametric censored least squares."""

import numpy as np
import pytest

from moirais.fn.sclst import sclst


@pytest.fixture()
def censored_data():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    t_true = 2.0 + 1.0 * x + rng.standard_normal(n) * 0.5
    c = rng.exponential(3, n)
    y = np.minimum(t_true, c)
    delta = (t_true <= c).astype(float)
    return y, X, delta


def test_returns_dict(censored_data):
    y, X, delta = censored_data
    result = sclst(y, X, delta)
    assert isinstance(result, dict)
    for k in ("coefficients", "se", "n_iter", "converged", "n_obs", "n_events"):
        assert k in result


def test_coefficients_shape(censored_data):
    y, X, delta = censored_data
    result = sclst(y, X, delta)
    assert result["coefficients"].shape == (2,)


def test_se_shape(censored_data):
    y, X, delta = censored_data
    result = sclst(y, X, delta)
    assert result["se"].shape == (2,)


def test_n_events_correct(censored_data):
    y, X, delta = censored_data
    result = sclst(y, X, delta)
    assert result["n_events"] == int(np.sum(delta))
    assert result["n_obs"] == len(y)


def test_all_uncensored():
    rng = np.random.default_rng(10)
    n = 50
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    y = 1.0 + 2.0 * x + rng.standard_normal(n) * 0.3
    delta = np.ones(n)
    result = sclst(y, X, delta)
    assert abs(result["coefficients"][1] - 2.0) < 1.0


def test_dimension_error():
    with pytest.raises(ValueError):
        sclst(np.array([1, 2]), np.array([[1], [2], [3]]), np.array([1, 1]))


def test_cheatsheet():
    from moirais.fn.sclst import cheatsheet
    assert "censored" in cheatsheet().lower()
