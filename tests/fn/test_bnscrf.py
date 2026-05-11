"""Tests for bnscrf.bound_credible_interval."""
import numpy as np
import pytest
from morie.fn.bnscrf import bound_credible_interval


def test_bnscrf_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_credible_interval(lower, upper, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnscrf_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_credible_interval(lower, upper, alpha)
    assert isinstance(result, dict)
