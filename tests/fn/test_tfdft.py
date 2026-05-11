"""Tests for tfdft.py - TFD feature extraction."""
import numpy as np
import pytest
from morie.fn.tfdft import tfdft_fn, tfdft


def test_tfdft_returns_descriptive_result():
    tfd = np.random.default_rng(42).random((32, 64))
    t = np.arange(64) / 64.0
    f = np.arange(32) * 0.5
    result = tfdft_fn(tfd, t, f)
    assert result.name == "tfd_features"
    assert "mean_time" in result.extra
    assert "mean_freq" in result.extra
    assert "total_energy" in result.extra


def test_tfdft_zero_tfd():
    tfd = np.zeros((16, 32))
    t = np.arange(32, dtype=float)
    f = np.arange(16, dtype=float)
    result = tfdft_fn(tfd, t, f)
    assert result.value == 0
    assert result.extra["mean_time"] == 0


def test_tfdft_positive_energy():
    tfd = np.ones((8, 16))
    t = np.arange(16, dtype=float)
    f = np.arange(8, dtype=float)
    result = tfdft_fn(tfd, t, f)
    assert result.value > 0


def test_tfdft_alias():
    tfd = np.random.default_rng(42).random((8, 16))
    t = np.arange(16, dtype=float)
    f = np.arange(8, dtype=float)
    result = tfdft(tfd, t, f)
    assert result.name == "tfd_features"
