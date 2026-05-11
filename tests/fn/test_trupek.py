"""Tests for trupek.trust_region."""
import numpy as np
import pytest
from morie.fn.trupek import trust_region


def test_trupek_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    hess_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = trust_region(f, grad_f, hess_f, x0, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trupek_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    hess_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = trust_region(f, grad_f, hess_f, x0, delta)
    assert isinstance(result, dict)
