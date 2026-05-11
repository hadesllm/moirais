"""Tests for klmnf.py - Kalman filter."""
import numpy as np
import pytest
from morie.fn.klmnf import kalman_fn, klmnf


def test_kalman_returns_signal_result():
    y = np.random.default_rng(42).standard_normal(100)
    result = kalman_fn(y)
    assert result.name == "kalman"
    assert result.filtered is not None
    assert result.n_samples == 100


def test_kalman_output_length():
    y = np.random.default_rng(42).standard_normal(100)
    result = kalman_fn(y)
    assert len(result.filtered) == 100


def test_kalman_smooths_signal():
    rng = np.random.default_rng(42)
    true_signal = np.linspace(0, 1, 100)
    y = true_signal + rng.standard_normal(100) * 0.5
    result = kalman_fn(y, Q=0.001, R=0.25)
    assert np.std(result.filtered) < np.std(y)


def test_klmnf_alias():
    y = np.random.default_rng(42).standard_normal(50)
    result = klmnf(y)
    assert result.name == "kalman"
