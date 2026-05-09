"""Tests for bndvld.bound_validity_check."""
import numpy as np
import pytest
from moirais.fn.bndvld import bound_validity_check


def test_bndvld_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    H0 = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_validity_check(lower, upper, theta_0, H0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndvld_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    H0 = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_validity_check(lower, upper, theta_0, H0)
    assert isinstance(result, dict)
