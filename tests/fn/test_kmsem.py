"""Tests for morie.fn.kmsem -- Kaplan-Meier survival estimator."""

import numpy as np
import pytest

from morie.fn.kmsem import kmsem


@pytest.fixture()
def survival_data():
    rng = np.random.default_rng(42)
    t = rng.exponential(5, 80)
    c = rng.exponential(8, 80)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return time, event


def test_returns_dict(survival_data):
    time, event = survival_data
    result = kmsem(time, event)
    assert isinstance(result, dict)
    for k in ("times", "survival", "se", "ci_lower", "ci_upper",
              "n_obs", "n_events", "median_survival"):
        assert k in result


def test_survival_monotone(survival_data):
    time, event = survival_data
    result = kmsem(time, event)
    diffs = np.diff(result["survival"])
    assert np.all(diffs <= 1e-12)


def test_survival_starts_near_one(survival_data):
    time, event = survival_data
    result = kmsem(time, event)
    assert result["survival"][0] <= 1.0


def test_se_nonnegative(survival_data):
    time, event = survival_data
    result = kmsem(time, event)
    assert np.all(result["se"] >= 0)


def test_ci_contains_survival(survival_data):
    time, event = survival_data
    result = kmsem(time, event)
    assert np.all(result["ci_lower"] <= result["survival"] + 1e-10)
    assert np.all(result["ci_upper"] >= result["survival"] - 1e-10)


def test_n_events_correct(survival_data):
    time, event = survival_data
    result = kmsem(time, event)
    assert result["n_events"] == int(np.sum(event))


def test_no_censoring():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.ones(5)
    result = kmsem(time, event)
    assert result["survival"][-1] == pytest.approx(0.0)


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        kmsem(np.array([]), np.array([]))


def test_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        kmsem(np.array([1, 2]), np.array([1]))


def test_cheatsheet():
    from morie.fn.kmsem import cheatsheet
    assert "kaplan" in cheatsheet().lower()
