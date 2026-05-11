"""Tests for morie.fn.strdp — Stratified propensity score."""
import numpy as np
import pytest
from morie.fn.strdp import strdp


@pytest.fixture()
def data():
    rng = np.random.default_rng(15)
    n = 300
    x = rng.standard_normal((n, 2))
    ps = 1 / (1 + np.exp(-(0.3 * x[:, 0])))
    t = rng.binomial(1, ps).astype(float)
    y = 2.0 * t + 0.4 * x[:, 0] + rng.standard_normal(n) * 0.4
    return y, t, x


def test_keys(data):
    r = strdp(*data)
    for k in ("ate", "se", "ci_lower", "ci_upper", "stratum_effects",
              "stratum_ns", "propensity", "n", "method"):
        assert k in r


def test_ate_close_to_truth(data):
    r = strdp(*data)
    assert abs(r["ate"] - 2.0) < 0.7


def test_stratum_count(data):
    r = strdp(*data, n_strata=5)
    assert len(r["stratum_effects"]) == 5


def test_propensity_shape(data):
    y, t, x = data
    r = strdp(*data)
    assert r["propensity"].shape == (len(y),)


def test_propensity_in_01(data):
    r = strdp(*data)
    assert np.all(r["propensity"] > 0) and np.all(r["propensity"] < 1)


def test_ci_valid(data):
    r = strdp(*data)
    assert r["ci_lower"] <= r["ate"] <= r["ci_upper"]


def test_method(data):
    r = strdp(*data, n_strata=5)
    assert "stratified-PS" in r["method"]


def test_cheatsheet():
    from morie.fn.strdp import cheatsheet
    assert len(cheatsheet()) > 0
