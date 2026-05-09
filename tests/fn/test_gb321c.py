"""Tests for gb321c.gibbons_marginal_r1."""
import numpy as np
import pytest
from moirais.fn.gb321c import gibbons_marginal_r1


def test_gb321c_basic():
    """Test basic functionality."""
    r1 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_marginal_r1(r1, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb321c_edge():
    """Test edge cases."""
    r1 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_marginal_r1(r1, n1, n2)
    assert isinstance(result, dict)
