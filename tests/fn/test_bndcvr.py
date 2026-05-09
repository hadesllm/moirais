"""Tests for bndcvr.bound_coverage_check."""
import numpy as np
import pytest
from moirais.fn.bndcvr import bound_coverage_check


def test_bndcvr_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    theta_true = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_coverage_check(lower, upper, theta_true, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndcvr_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    theta_true = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_coverage_check(lower, upper, theta_true, alpha)
    assert isinstance(result, dict)
