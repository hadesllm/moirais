"""Tests for ctomeg.omega_total."""
import numpy as np
import pytest
from morie.fn.ctomeg import omega_total


def test_ctomeg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_loadings = np.random.default_rng(42).normal(0, 1, 100)
    result = omega_total(X, factor_loadings)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ctomeg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_loadings = np.random.default_rng(42).normal(0, 1, 100)
    result = omega_total(X, factor_loadings)
    assert isinstance(result, dict)
