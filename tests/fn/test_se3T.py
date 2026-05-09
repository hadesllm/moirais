"""Tests for se3T.se3_transformer."""
import numpy as np
import pytest
from moirais.fn.se3T import se3_transformer


def test_se3T_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = se3_transformer(G, X, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_se3T_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = se3_transformer(G, X, coords)
    assert isinstance(result, dict)
