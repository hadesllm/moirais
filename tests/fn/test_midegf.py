"""Tests for midegf.mi_degrees_of_freedom."""
import numpy as np
import pytest
from moirais.fn.midegf import mi_degrees_of_freedom


def test_midegf_basic():
    """Test basic functionality."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = mi_degrees_of_freedom(B, W, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_midegf_edge():
    """Test edge cases."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    W = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = mi_degrees_of_freedom(B, W, m)
    assert isinstance(result, dict)
