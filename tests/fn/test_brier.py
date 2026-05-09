"""Tests for brier: Brier score for survival models."""
import numpy as np
import pytest
from moirais.fn.brier import brier


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.exponential(2.0, size=n)
    C = rng.exponential(4.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_returns_keys():
    time, event = _make_data()
    # Predict S(t=1) = 0.6 for all subjects
    pred = np.full(len(time), 0.6)
    result = brier(time, event, pred, eval_time=1.0)
    for key in ("brier_score", "scaled_brier", "integrated_brier", "eval_time"):
        assert key in result


def test_brier_between_0_and_1():
    time, event = _make_data()
    pred = np.full(len(time), 0.5)
    result = brier(time, event, pred, eval_time=1.0)
    assert 0.0 <= result["brier_score"] <= 1.0


def test_perfect_prediction_low_brier():
    """Perfect binary prediction should give lower Brier score than random."""
    rng = np.random.default_rng(42)
    n = 500
    T = rng.exponential(2.0, size=n)
    C = rng.exponential(6.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    tau = 2.0
    # Perfect: S(tau|x) = 1 if T > tau, else 0
    pred_perfect = (time > tau).astype(float)
    pred_random = np.full(n, 0.5)
    r_perfect = brier(time, event, pred_perfect, eval_time=tau)
    r_random = brier(time, event, pred_random, eval_time=tau)
    assert r_perfect["brier_score"] <= r_random["brier_score"]


def test_naive_method():
    time, event = _make_data()
    pred = np.full(len(time), 0.6)
    result = brier(time, event, pred, eval_time=1.0, method="naive")
    assert 0.0 <= result["brier_score"] <= 1.0


def test_multiple_eval_times():
    time, event = _make_data()
    n = len(time)
    eval_times = np.array([0.5, 1.0, 2.0, 3.0])
    pred_2d = np.full((n, 4), 0.5)
    result = brier(time, event, pred_2d, eval_time=eval_times)
    assert len(result["brier_score"]) == 4
    assert np.isfinite(result["integrated_brier"])


def test_invalid_method_raises():
    time, event = _make_data()
    pred = np.full(len(time), 0.5)
    with pytest.raises(ValueError):
        brier(time, event, pred, eval_time=1.0, method="bogus")


def test_scaled_brier_leq_1():
    time, event = _make_data()
    pred = np.full(len(time), 0.5)
    result = brier(time, event, pred, eval_time=1.0)
    assert result["scaled_brier"] <= 1.0


def test_brier_score_finite():
    time, event = _make_data()
    pred = np.random.default_rng(7).uniform(0.1, 0.9, size=len(time))
    result = brier(time, event, pred, eval_time=1.5)
    assert np.isfinite(result["brier_score"])
