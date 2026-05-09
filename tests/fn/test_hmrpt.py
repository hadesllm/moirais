"""Tests for hmrpt.geron_random_patches."""
import numpy as np
import pytest
from moirais.fn.hmrpt import geron_random_patches


def test_hmrpt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_samples = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_random_patches(X, y, base_estimator, n_estimators, max_samples, max_features)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrpt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    n_estimators = np.random.default_rng(42).normal(0, 1, 100)
    max_samples = np.random.default_rng(42).normal(0, 1, 100)
    max_features = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_random_patches(X, y, base_estimator, n_estimators, max_samples, max_features)
    assert isinstance(result, dict)
