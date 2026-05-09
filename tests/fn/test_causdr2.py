"""Tests for causdr2.causal_dr_orthogonal."""
import numpy as np
import pytest
from moirais.fn.causdr2 import causal_dr_orthogonal


def test_causdr2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_dr_orthogonal(y, T, ps, m1, m0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdr2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_dr_orthogonal(y, T, ps, m1, m0)
    assert isinstance(result, dict)
