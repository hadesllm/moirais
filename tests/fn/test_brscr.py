"""Tests for moirais.fn.brscr -- Brier score for survival."""

import numpy as np
import pytest

from moirais.fn.brscr import brscr


@pytest.fixture()
def bs_data():
    rng = np.random.default_rng(42)
    n = 50
    time = rng.exponential(5, n)
    event = rng.binomial(1, 0.7, n).astype(float)
    pred_surv = np.exp(-time / 5)
    return pred_surv, time, event


def test_returns_dict(bs_data):
    pred, time, event = bs_data
    r = brscr(pred, time, event, eval_time=3.0)
    assert isinstance(r, dict)
    for k in ("brier_score", "ipcw_brier_score", "eval_time"):
        assert k in r


def test_brier_bounded(bs_data):
    pred, time, event = bs_data
    r = brscr(pred, time, event, eval_time=3.0)
    assert 0 <= r["brier_score"] <= 1


def test_perfect_prediction():
    time = np.array([1, 2, 3, 4, 5.0])
    event = np.ones(5)
    pred = np.array([0, 0, 0, 1, 1.0])
    r = brscr(pred, time, event, eval_time=3.5)
    assert r["brier_score"] < 0.5


def test_cheatsheet():
    from moirais.fn.brscr import cheatsheet
    assert "brier" in cheatsheet().lower()
