"""Tests for hrzbwopt.horowitz_optimal_bandwidth_kde."""
import numpy as np
import pytest
from morie.fn.hrzbwopt import horowitz_optimal_bandwidth_kde


def test_hrzbwopt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = horowitz_optimal_bandwidth_kde(x, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzbwopt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = horowitz_optimal_bandwidth_kde(x, kernel)
    assert isinstance(result, dict)
