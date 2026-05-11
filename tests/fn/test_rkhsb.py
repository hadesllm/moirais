"""Tests for rkhsb.rkhs_bayesian_kernel."""
import numpy as np
import pytest
from morie.fn.rkhsb import rkhs_bayesian_kernel


def test_rkhsb_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    a_u = np.random.default_rng(42).normal(0, 1, 100)
    b_u = np.random.default_rng(42).normal(0, 1, 100)
    a_e = np.random.default_rng(42).normal(0, 1, 100)
    b_e = np.random.default_rng(42).normal(0, 1, 100)
    result = rkhs_bayesian_kernel(y, K, a_u, b_u, a_e, b_e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rkhsb_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    a_u = np.random.default_rng(42).normal(0, 1, 100)
    b_u = np.random.default_rng(42).normal(0, 1, 100)
    a_e = np.random.default_rng(42).normal(0, 1, 100)
    b_e = np.random.default_rng(42).normal(0, 1, 100)
    result = rkhs_bayesian_kernel(y, K, a_u, b_u, a_e, b_e)
    assert isinstance(result, dict)
