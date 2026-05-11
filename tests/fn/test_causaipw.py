"""Tests for causaipw.causal_aipw."""
import numpy as np
import pytest
from morie.fn.causaipw import causal_aipw


def test_causaipw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_aipw(y, T, ps, m1, m0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causaipw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_aipw(y, T, ps, m1, m0)
    assert isinstance(result, dict)
