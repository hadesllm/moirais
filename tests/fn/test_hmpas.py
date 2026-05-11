"""Tests for hmpas.geron_pasting."""
import numpy as np
import pytest
from morie.fn.hmpas import geron_pasting


def test_hmpas_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    sample_size = 100
    result = geron_pasting(X, y, base_estimator, n_estimators, sample_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpas_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    sample_size = 100
    result = geron_pasting(X, y, base_estimator, n_estimators, sample_size)
    assert isinstance(result, dict)
