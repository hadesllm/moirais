"""Tests for morie.fn.rgknd — Regression kink design."""
import numpy as np
import pytest
from morie.fn.rgknd import rgknd


@pytest.fixture()
def data():
    rng = np.random.default_rng(13)
    n = 400
    R = rng.uniform(-2, 2, n)
    T = np.where(R < 0, 0.5 * R, 1.5 * R)   # kink at R=0
    true_tau = 2.0
    Y = true_tau * T + 0.3 * R + rng.standard_normal(n) * 0.3
    return Y, T, R


def test_keys(data):
    r = rgknd(*data)
    for k in ("tau", "se", "ci_lower", "ci_upper", "n_used", "bandwidth", "method"):
        assert k in r


def test_tau_finite(data):
    r = rgknd(*data)
    assert np.isfinite(r["tau"])


def test_bandwidth_positive(data):
    r = rgknd(*data)
    assert r["bandwidth"] > 0


def test_n_used_leq_n(data):
    y, t, r = data
    res = rgknd(*data)
    assert res["n_used"] <= len(y)


def test_ci_valid(data):
    r = rgknd(*data)
    assert r["ci_lower"] <= r["tau"] <= r["ci_upper"]


def test_method(data):
    assert rgknd(*data)["method"] == "RKD"


def test_slope_signs(data):
    r = rgknd(*data)
    # Treatment slope increases at cutoff
    assert r["slope_T_right"] > r["slope_T_left"]


def test_cheatsheet():
    from morie.fn.rgknd import cheatsheet
    assert len(cheatsheet()) > 0
