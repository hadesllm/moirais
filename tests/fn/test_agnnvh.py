"""Tests for agnnvh.alphazero_value_head."""
import numpy as np
import pytest
from morie.fn.agnnvh import alphazero_value_head


def test_agnnvh_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    theta = 0.0
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_value_head(z, v, pi, p, theta, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agnnvh_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    theta = 0.0
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_value_head(z, v, pi, p, theta, c)
    assert isinstance(result, dict)
