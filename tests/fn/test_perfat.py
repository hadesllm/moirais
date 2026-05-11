"""Tests for perfat.performer_favor_attention."""
import numpy as np
import pytest
from morie.fn.perfat import performer_favor_attention


def test_perfat_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = performer_favor_attention(y, Q, K, V, phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_perfat_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    result = performer_favor_attention(y, Q, K, V, phi)
    assert isinstance(result, dict)
