"""Tests for morie.fn.strtf -- Stratified Cox model."""

import numpy as np
import pytest

from morie.fn.strtf import strtf


@pytest.fixture()
def strat_data():
    rng = np.random.default_rng(42)
    n = 120
    X = rng.standard_normal((n, 2))
    strata = rng.choice([0, 1], n)
    lp = 0.5 * X[:, 0]
    t = rng.exponential(np.exp(-lp))
    c = rng.exponential(3, n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event, X, strata


def test_returns_dict(strat_data):
    time, event, X, strata = strat_data
    r = strtf(time, event, X, strata)
    assert isinstance(r, dict)
    for k in ("coefficients", "se", "hazard_ratios", "n_strata"):
        assert k in r


def test_coefficients_shape(strat_data):
    time, event, X, strata = strat_data
    r = strtf(time, event, X, strata)
    assert r["coefficients"].shape == (2,)


def test_n_strata(strat_data):
    time, event, X, strata = strat_data
    r = strtf(time, event, X, strata)
    assert r["n_strata"] == 2


def test_cheatsheet():
    from morie.fn.strtf import cheatsheet
    assert "stratified" in cheatsheet().lower()
