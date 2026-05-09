"""Tests for medfl -- Median filter."""
import numpy as np
from moirais.fn.medfl import medfl
from moirais.fn._containers import SignalResult


def test_medfl_basic():
    x = np.array([1.0, 2.0, 100.0, 3.0, 4.0])
    result = medfl(x, kernel_size=3)
    assert isinstance(result, SignalResult)
    assert result.n_samples == 5


def test_medfl_removes_spike():
    x = np.ones(100)
    x[50] = 100.0
    result = medfl(x, kernel_size=5)
    assert result.filtered[50] < 10.0


def test_medfl_even_kernel():
    x = np.random.default_rng(42).standard_normal(200)
    result = medfl(x, kernel_size=4)
    assert result.extra["kernel_size"] == 5


def test_medfl_preserves_length():
    x = np.random.default_rng(7).standard_normal(300)
    result = medfl(x)
    assert len(result.filtered) == 300
