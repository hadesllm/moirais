"""Tests for acfds.py - ACF distance between signals."""
import numpy as np
import pytest
from morie.fn.acfds import acf_dist, acfds


def test_acf_dist_returns_descriptive_result():
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(256)
    x2 = rng.standard_normal(256)
    result = acf_dist(x1, x2)
    assert result.name == "acf_distance"
    assert isinstance(result.value, float)


def test_acf_dist_identical_signals_zero():
    x = np.random.default_rng(42).standard_normal(256)
    result = acf_dist(x, x)
    assert result.value == pytest.approx(0.0, abs=1e-10)


def test_acf_dist_nonnegative():
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(256)
    x2 = rng.standard_normal(256)
    result = acf_dist(x1, x2)
    assert result.value >= 0


def test_acfds_alias():
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal(64)
    x2 = rng.standard_normal(64)
    result = acfds(x1, x2, max_lag=10)
    assert result.name == "acf_distance"
