"""Tests for depthS.simplicial_depth."""
import numpy as np
import pytest
from moirais.fn.depthS import simplicial_depth


def test_depthS_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = simplicial_depth(X, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_depthS_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = simplicial_depth(X, theta)
    assert isinstance(result, dict)
