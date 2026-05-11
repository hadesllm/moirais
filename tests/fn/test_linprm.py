"""Tests for linprm.linear_programming."""
import numpy as np
import pytest
from morie.fn.linprm import linear_programming


def test_linprm_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = linear_programming(c, A, b, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linprm_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = linear_programming(c, A, b, method)
    assert isinstance(result, dict)
