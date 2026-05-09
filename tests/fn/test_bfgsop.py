"""Tests for bfgsop.bfgs."""
import numpy as np
import pytest
from moirais.fn.bfgsop import bfgs


def test_bfgsop_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = bfgs(f, grad_f, x0, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bfgsop_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = bfgs(f, grad_f, x0, steps)
    assert isinstance(result, dict)
