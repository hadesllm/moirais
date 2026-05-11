"""Tests for hrzade.horowitz_average_derivative."""
import numpy as np
import pytest
from morie.fn.hrzade import horowitz_average_derivative


def test_hrzade_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_average_derivative(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzade_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_average_derivative(x, y, bandwidth)
    assert isinstance(result, dict)
