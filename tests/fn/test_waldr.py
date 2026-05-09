"""Tests for waldr.wald_estimator."""
import numpy as np
import pytest
from moirais.fn.waldr import wald_estimator


def test_waldr_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = wald_estimator(Y, X, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_waldr_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = wald_estimator(Y, X, Z)
    assert isinstance(result, dict)
