"""Tests for morie.fn.hazard — Hazard rate."""
import numpy as np

from morie.fn.hazard import hazard_rate, hazard


def test_hazard_nonnegative():
    """Hazard rate should be non-negative everywhere."""
    rng = np.random.default_rng(42)
    time = rng.exponential(5, size=100)
    event = np.ones(100, dtype=int)
    result = hazard_rate(time, event)
    assert np.all(result.survival >= 0)  # survival field holds hazard values


def test_hazard_has_grid():
    """Should return a grid of time points."""
    rng = np.random.default_rng(42)
    time = rng.exponential(5, size=100)
    event = rng.integers(0, 2, size=100)
    result = hazard_rate(time, event)
    assert len(result.times) == 100
    assert result.n_events == int(event.sum())


def test_hazard_alias():
    assert hazard is hazard_rate
