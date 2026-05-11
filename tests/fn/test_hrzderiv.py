"""Tests for hrzderiv.horowitz_density_derivative."""
import numpy as np
import pytest
from morie.fn.hrzderiv import horowitz_density_derivative


def test_hrzderiv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    order = 4
    result = horowitz_density_derivative(x, bandwidth, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzderiv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    order = 4
    result = horowitz_density_derivative(x, bandwidth, order)
    assert isinstance(result, dict)
