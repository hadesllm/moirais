"""Tests for otunbal.ot_unbalanced."""
import numpy as np
import pytest
from morie.fn.otunbal import ot_unbalanced


def test_otunbal_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    lam = 0.1
    result = ot_unbalanced(a, b, C, epsilon, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otunbal_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    lam = 0.1
    result = ot_unbalanced(a, b, C, epsilon, lam)
    assert isinstance(result, dict)
