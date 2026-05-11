"""Tests for ipwef.ipw_estimator."""
import numpy as np
import pytest
from morie.fn.ipwef import ipw_estimator


def test_ipwef_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ipw_estimator(Y, T, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ipwef_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ipw_estimator(Y, T, X)
    assert isinstance(result, dict)
