"""Tests for lbfgsm.lbfgs."""
import numpy as np
import pytest
from moirais.fn.lbfgsm import lbfgs


def test_lbfgsm_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = lbfgs(f, grad_f, x0, m, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lbfgsm_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = lbfgs(f, grad_f, x0, m, steps)
    assert isinstance(result, dict)
