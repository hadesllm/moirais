"""Tests for morie.fn.recur -- Recurrent event (Andersen-Gill)."""

import numpy as np
import pytest

from morie.fn.recur import recur


@pytest.fixture()
def recur_data():
    rng = np.random.default_rng(42)
    n = 80
    start = np.zeros(n)
    stop = rng.exponential(5, n)
    event = rng.binomial(1, 0.6, n).astype(float)
    X = rng.standard_normal((n, 2))
    return start, stop, event, X


def test_returns_dict(recur_data):
    start, stop, event, X = recur_data
    r = recur(start, stop, event, X)
    assert isinstance(r, dict)
    for k in ("coefficients", "se", "hazard_ratios", "converged"):
        assert k in r


def test_coefficients_shape(recur_data):
    start, stop, event, X = recur_data
    r = recur(start, stop, event, X)
    assert r["coefficients"].shape == (2,)


def test_hr_positive(recur_data):
    start, stop, event, X = recur_data
    r = recur(start, stop, event, X)
    assert np.all(r["hazard_ratios"] > 0)


def test_cheatsheet():
    from morie.fn.recur import cheatsheet
    assert "andersen" in cheatsheet().lower()
