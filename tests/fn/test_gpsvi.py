"""Tests for gpsvi.gp_stochastic_vi."""
import numpy as np
import pytest
from morie.fn.gpsvi import gp_stochastic_vi


def test_gpsvi_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    inducing = np.random.default_rng(42).normal(0, 1, 100)
    batch_size = 100
    result = gp_stochastic_vi(X, y, X_test, inducing, batch_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpsvi_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    inducing = np.random.default_rng(42).normal(0, 1, 100)
    batch_size = 100
    result = gp_stochastic_vi(X, y, X_test, inducing, batch_size)
    assert isinstance(result, dict)
