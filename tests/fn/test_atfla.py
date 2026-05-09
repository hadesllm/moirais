"""I cannot teach anybody anything. I can only make them think. — Socrates"""
import numpy as np
import pytest
from moirais.fn.atfla import flash_attention_block


def test_atfla_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    block_size = 100
    result = flash_attention_block(y, Q, K, V, block_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_atfla_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    block_size = 100
    result = flash_attention_block(y, Q, K, V, block_size)
    assert isinstance(result, dict)
