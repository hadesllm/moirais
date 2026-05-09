"""Without music, life would be a mistake. — Friedrich Nietzsche"""
import numpy as np
import pytest
from moirais.fn.hmfa import geron_flash_attention


def test_hmfa_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    block_size = 100
    result = geron_flash_attention(Q, K, V, block_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfa_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    block_size = 100
    result = geron_flash_attention(Q, K, V, block_size)
    assert isinstance(result, dict)
