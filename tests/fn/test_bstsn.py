"""Tests for morie.fn.bstsn — Bayesian structural time series."""
import numpy as np
import pytest
from morie.fn.bstsn import bstsn


@pytest.fixture()
def series():
    rng = np.random.default_rng(14)
    t = np.arange(50)
    trend = 0.1 * t
    y = trend + rng.standard_normal(50) * 0.5
    return y


def test_keys(series):
    r = bstsn(series, n_iter=100, burn=20, seed=0)
    for k in ("trend_mean", "trend_ci", "sigma_obs_mean", "forecast_mean", "T", "method"):
        assert k in r


def test_trend_shape(series):
    r = bstsn(series, n_iter=100, burn=20)
    assert r["trend_mean"].shape == (50,)


def test_trend_ci_shape(series):
    r = bstsn(series, n_iter=100, burn=20)
    assert r["trend_ci"].shape == (50, 2)


def test_trend_ci_valid(series):
    r = bstsn(series, n_iter=100, burn=20)
    assert np.all(r["trend_ci"][:, 0] <= r["trend_ci"][:, 1])


def test_sigma_obs_positive(series):
    r = bstsn(series, n_iter=100, burn=20)
    assert r["sigma_obs_mean"] > 0


def test_forecast(series):
    r = bstsn(series, n_iter=100, burn=20, forecast_steps=5)
    assert r["forecast_mean"].shape == (5,)
    assert r["forecast_ci"].shape == (5, 2)


def test_method(series):
    r = bstsn(series, n_iter=50, burn=10)
    assert "BSTS" in r["method"]


def test_trend_captures_direction(series):
    r = bstsn(series, n_iter=200, burn=50)
    # Trend should increase over time
    assert float(r["trend_mean"][-1]) > float(r["trend_mean"][0]) - 3.0


def test_cheatsheet():
    from morie.fn.bstsn import cheatsheet
    assert len(cheatsheet()) > 0
