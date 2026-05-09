"""Tests for kalmf -- Kalman filter."""
import numpy as np
from moirais.fn.kalmf import kalmf
from moirais.fn._containers import SignalResult


def test_kalmf_basic():
    rng = np.random.default_rng(42)
    x = np.ones(100) + rng.standard_normal(100) * 0.5
    result = kalmf(x)
    assert isinstance(result, SignalResult)
    assert result.n_samples == 100


def test_kalmf_reduces_noise():
    rng = np.random.default_rng(42)
    true_val = 5.0
    x = np.full(200, true_val) + rng.standard_normal(200) * 1.0
    result = kalmf(x, R=1.0, Q=0.01)
    assert abs(result.filtered[-1] - true_val) < 0.5


def test_kalmf_gain_decreases():
    x = np.random.default_rng(7).standard_normal(100)
    result = kalmf(x, R=1.0, Q=0.001)
    K = result.extra["kalman_gain"]
    assert K[-1] < K[0]
