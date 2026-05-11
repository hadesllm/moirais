"""Tests for cohfn.py - Coherence function."""
import numpy as np
from morie.fn.cohfn import coherence_function_fn, cohfn


def test_cohfn_returns_result():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    y = x + 0.1 * rng.standard_normal(512)
    result = coherence_function_fn(x, y, fs=100.0, nperseg=128)
    assert result.name == "coherence_function"
    assert "coherence" in result.extra
    assert "frequencies" in result.extra


def test_cohfn_identical_signals():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = coherence_function_fn(x, x, fs=100.0, nperseg=128)
    assert np.all(result.extra["coherence"] > 0.99)


def test_cohfn_range():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    y = rng.standard_normal(512)
    result = coherence_function_fn(x, y, fs=100.0, nperseg=128)
    assert np.all(result.extra["coherence"] >= 0)
    assert np.all(result.extra["coherence"] <= 1)


def test_cohfn_alias():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(256)
    y = rng.standard_normal(256)
    result = cohfn(x, y, nperseg=64)
    assert result.name == "coherence_function"
