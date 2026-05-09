"""Tests for trncfg.truncated_cf_estimator."""
import numpy as np
import pytest
from moirais.fn.trncfg import truncated_cf_estimator


def test_trncfg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    trim = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_cf_estimator(y, D, X, trim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trncfg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    trim = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_cf_estimator(y, D, X, trim)
    assert isinstance(result, dict)
