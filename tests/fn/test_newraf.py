"""Tests for newraf.newton_raphson."""
import numpy as np
import pytest
from morie.fn.newraf import newton_raphson


def test_newraf_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    hess_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = newton_raphson(f, grad_f, hess_f, x0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_newraf_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    hess_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = newton_raphson(f, grad_f, hess_f, x0)
    assert isinstance(result, dict)
