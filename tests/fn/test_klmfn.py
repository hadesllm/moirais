"""Tests for klmfn.py - Kalman filter wrapper."""
import numpy as np
from morie.fn.klmfn import kalman_filter_fn, klmfn


def test_klmfn_returns_signal_result():
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.standard_normal(100)) + rng.standard_normal(100)
    result = kalman_filter_fn(y)
    assert result.name == "kalman_filter"
    assert result.n_samples == 100


def test_klmfn_smooths_noisy_signal():
    rng = np.random.default_rng(42)
    true_signal = np.sin(np.linspace(0, 4 * np.pi, 200))
    y = true_signal + 0.5 * rng.standard_normal(200)
    result = kalman_filter_fn(y, Q=0.01, R=0.25)
    noise_var = np.var(y - true_signal)
    filtered_var = np.var(result.filtered - true_signal)
    assert filtered_var < noise_var


def test_klmfn_extra_fields():
    y = np.random.default_rng(42).standard_normal(50)
    result = kalman_filter_fn(y)
    assert "state_estimates" in result.extra
    assert "error_covariance" in result.extra


def test_klmfn_alias():
    y = np.random.default_rng(42).standard_normal(30)
    result = klmfn(y)
    assert result.name == "kalman_filter"
