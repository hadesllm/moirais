"""Tests for attsdp.scaled_dot_product_attention."""
import numpy as np
import pytest
from morie.fn.attsdp import scaled_dot_product_attention


def test_attsdp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = scaled_dot_product_attention(y, Q, K, V, mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_attsdp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = scaled_dot_product_attention(y, Q, K, V, mask)
    assert isinstance(result, dict)
