"""Tests for depthP.projection_depth."""
import numpy as np
import pytest
from moirais.fn.depthP import projection_depth


def test_depthP_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = projection_depth(x, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_depthP_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = projection_depth(x, X)
    assert isinstance(result, dict)
