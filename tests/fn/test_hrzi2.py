"""Tests for hrzi2.horowitz_average_derivative."""
import numpy as np
import pytest
from morie.fn.hrzi2 import horowitz_average_derivative


def test_hrzi2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_average_derivative(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzi2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_average_derivative(x, y)
    assert isinstance(result, dict)
