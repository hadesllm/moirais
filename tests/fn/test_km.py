"""Tests for morie.fn.km — Kaplan-Meier estimator."""
import numpy as np

from morie.fn.km import kaplan_meier, km


def test_km_starts_at_one():
    """Survival should start at 1.0."""
    rng = np.random.default_rng(42)
    time = rng.exponential(5, size=100)
    event = rng.integers(0, 2, size=100)
    result = kaplan_meier(time, event)
    assert result.survival[0] == 1.0
    assert result.times[0] == 0.0


def test_km_decreasing():
    """Survival curve should be non-increasing."""
    rng = np.random.default_rng(42)
    time = rng.exponential(5, size=100)
    event = np.ones(100, dtype=int)
    result = kaplan_meier(time, event)
    diffs = np.diff(result.survival)
    assert np.all(diffs <= 1e-10)


def test_km_all_events():
    """With all events, survival should reach near zero."""
    time = np.array([1, 2, 3, 4, 5], dtype=float)
    event = np.array([1, 1, 1, 1, 1])
    result = kaplan_meier(time, event)
    assert result.survival[-1] == 0.0
    assert result.n_events == 5
    assert result.n_censored == 0


def test_km_alias():
    assert km is kaplan_meier
