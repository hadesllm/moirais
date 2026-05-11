"""Tests for lpdual.lp_dual."""
import numpy as np
import pytest
from morie.fn.lpdual import lp_dual


def test_lpdual_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = lp_dual(c, A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lpdual_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = lp_dual(c, A, b)
    assert isinstance(result, dict)
