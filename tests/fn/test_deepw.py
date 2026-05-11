"""Tests for deepw.deepwalk."""
import numpy as np
import pytest
from morie.fn.deepw import deepwalk


def test_deepw_basic():
    """Test basic functionality."""
    G = np.eye(10)
    walk_len = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = deepwalk(G, walk_len, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deepw_edge():
    """Test edge cases."""
    G = np.eye(10)
    walk_len = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = deepwalk(G, walk_len, dim)
    assert isinstance(result, dict)
