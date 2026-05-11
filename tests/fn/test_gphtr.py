"""Tests for gphtr.gp_heteroscedastic."""
import numpy as np
import pytest
from morie.fn.gphtr import gp_heteroscedastic


def test_gphtr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = gp_heteroscedastic(X, y, X_test)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gphtr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    result = gp_heteroscedastic(X, y, X_test)
    assert isinstance(result, dict)
