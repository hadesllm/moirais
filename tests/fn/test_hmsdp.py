"""Tests for hmsdp.geron_scaled_dot_product."""
import numpy as np
import pytest
from morie.fn.hmsdp import geron_scaled_dot_product


def test_hmsdp_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_scaled_dot_product(Q, K, V, d_k, mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsdp_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    d_k = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_scaled_dot_product(Q, K, V, d_k, mask)
    assert isinstance(result, dict)
