"""Tests for moirais.fn.bartk — Bartik/shift-share IV."""
import numpy as np
import pytest
from moirais.fn.bartk import bartk


@pytest.fixture()
def data():
    rng = np.random.default_rng(12)
    L, K = 80, 5
    shares = rng.dirichlet(np.ones(K), size=L)
    shifts = rng.standard_normal(K)
    Z = shares @ shifts
    Y = 1.5 * Z + rng.standard_normal(L) * 0.5
    return Y, shares, shifts


def test_keys(data):
    r = bartk(*data)
    for k in ("beta_2sls", "se", "ci_lower", "ci_upper", "first_stage_f",
              "instrument", "L", "K", "method"):
        assert k in r


def test_instrument_shape(data):
    y, shares, shifts = data
    r = bartk(*data)
    assert r["instrument"].shape == (len(y),)


def test_L_K_correct(data):
    y, shares, shifts = data
    r = bartk(*data)
    assert r["L"] == len(y)
    assert r["K"] == len(shifts)


def test_beta_finite(data):
    r = bartk(*data)
    assert np.isfinite(r["beta_2sls"])


def test_first_stage_positive(data):
    r = bartk(*data)
    assert r["first_stage_f"] > 0


def test_ci_valid(data):
    r = bartk(*data)
    assert r["ci_lower"] <= r["beta_2sls"] <= r["ci_upper"]


def test_method(data):
    assert bartk(*data)["method"] == "Bartik-IV"


def test_cheatsheet():
    from moirais.fn.bartk import cheatsheet
    assert len(cheatsheet()) > 0
