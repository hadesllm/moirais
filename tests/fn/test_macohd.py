"""Tests for macohd.ma_cohens_d."""
import numpy as np
import pytest
from morie.fn.macohd import ma_cohens_d


def test_macohd_basic():
    """Test basic functionality."""
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m2 = np.random.default_rng(42).normal(0, 1, 100)
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_cohens_d(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_macohd_edge():
    """Test edge cases."""
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m2 = np.random.default_rng(42).normal(0, 1, 100)
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_cohens_d(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
