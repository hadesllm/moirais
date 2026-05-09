"""Tests for kmalbi.kamath_alibi_bias."""
import numpy as np
import pytest
from moirais.fn.kmalbi import kamath_alibi_bias


def test_kmalbi_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    slopes = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_alibi_bias(Q, K, V, slopes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmalbi_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    slopes = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_alibi_bias(Q, K, V, slopes)
    assert isinstance(result, dict)
