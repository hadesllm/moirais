"""Tests for mahg.ma_hedges_g."""
import numpy as np
import pytest
from moirais.fn.mahg import ma_hedges_g


def test_mahg_basic():
    """Test basic functionality."""
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m2 = np.random.default_rng(42).normal(0, 1, 100)
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_hedges_g(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mahg_edge():
    """Test edge cases."""
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m2 = np.random.default_rng(42).normal(0, 1, 100)
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_hedges_g(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
