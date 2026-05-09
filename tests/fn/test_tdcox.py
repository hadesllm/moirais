"""Tests for moirais.fn.tdcox -- Time-dependent covariates Cox."""

import numpy as np
import pytest

from moirais.fn.tdcox import tdcox


@pytest.fixture()
def td_data():
    rng = np.random.default_rng(42)
    n = 80
    start = np.zeros(n)
    stop = rng.exponential(5, n)
    event = rng.binomial(1, 0.6, n).astype(float)
    X = rng.standard_normal((n, 2))
    return start, stop, event, X


def test_returns_dict(td_data):
    start, stop, event, X = td_data
    r = tdcox(start, stop, event, X)
    assert isinstance(r, dict)
    for k in ("coefficients", "se", "hazard_ratios", "p_values", "converged"):
        assert k in r


def test_coefficients_shape(td_data):
    start, stop, event, X = td_data
    r = tdcox(start, stop, event, X)
    assert r["coefficients"].shape == (2,)


def test_hr_positive(td_data):
    start, stop, event, X = td_data
    r = tdcox(start, stop, event, X)
    assert np.all(r["hazard_ratios"] > 0)


def test_cheatsheet():
    from moirais.fn.tdcox import cheatsheet
    assert "time-dependent" in cheatsheet().lower()
