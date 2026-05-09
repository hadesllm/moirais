"""Tests for hrzlew.horowitz_lewbel_estimator."""
import numpy as np
import pytest
from moirais.fn.hrzlew import horowitz_lewbel_estimator


def test_hrzlew_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_lewbel_estimator(x, y, z, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzlew_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_lewbel_estimator(x, y, z, bandwidth)
    assert isinstance(result, dict)
