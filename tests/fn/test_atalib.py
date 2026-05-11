"""Tests for atalib.alibi_position_bias."""
import numpy as np
import pytest
from morie.fn.atalib import alibi_position_bias


def test_atalib_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    slopes = np.random.default_rng(42).normal(0, 1, 100)
    result = alibi_position_bias(y, Q, K, V, slopes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_atalib_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    slopes = np.random.default_rng(42).normal(0, 1, 100)
    result = alibi_position_bias(y, Q, K, V, slopes)
    assert isinstance(result, dict)
