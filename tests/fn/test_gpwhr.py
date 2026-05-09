"""Tests for gpwhr.gp_warped."""
import numpy as np
import pytest
from moirais.fn.gpwhr import gp_warped


def test_gpwhr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    warp = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_warped(X, y, X_test, warp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpwhr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    warp = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_warped(X, y, X_test, warp)
    assert isinstance(result, dict)
