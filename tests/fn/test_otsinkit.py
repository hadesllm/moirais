"""Tests for otsinkit.ot_sinkhorn_iter_count."""
import numpy as np
import pytest
from morie.fn.otsinkit import ot_sinkhorn_iter_count


def test_otsinkit_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    tol = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sinkhorn_iter_count(a, b, C, epsilon, tol, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otsinkit_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    tol = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sinkhorn_iter_count(a, b, C, epsilon, tol, max_iter)
    assert isinstance(result, dict)
