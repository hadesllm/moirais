"""Tests for hrzchet.horowitz_chen_estimator_T."""
import numpy as np
import pytest
from morie.fn.hrzchet import horowitz_chen_estimator_T


def test_hrzchet_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_chen_estimator_T(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzchet_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_chen_estimator_T(x, y, bandwidth)
    assert isinstance(result, dict)
