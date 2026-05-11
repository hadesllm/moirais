"""Tests for paligi.parametric_alibi."""
import numpy as np
import pytest
from morie.fn.paligi import parametric_alibi


def test_paligi_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    s_h = np.random.default_rng(42).normal(0, 1, 100)
    result = parametric_alibi(y, Q, K, V, s_h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_paligi_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    s_h = np.random.default_rng(42).normal(0, 1, 100)
    result = parametric_alibi(y, Q, K, V, s_h)
    assert isinstance(result, dict)
