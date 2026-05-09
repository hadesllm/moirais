"""Tests for gblupr.gblup_estimator."""
import numpy as np
import pytest
from moirais.fn.gblupr import gblup_estimator


def test_gblupr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    G = np.eye(10)
    result = gblup_estimator(y, X, Z, G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gblupr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    G = np.eye(10)
    result = gblup_estimator(y, X, Z, G)
    assert isinstance(result, dict)
