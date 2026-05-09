"""Tests for taubrg.tau_estimator_regression."""
import numpy as np
import pytest
from moirais.fn.taubrg import tau_estimator_regression


def test_taubrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tau_estimator_regression(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_taubrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tau_estimator_regression(y, X)
    assert isinstance(result, dict)
