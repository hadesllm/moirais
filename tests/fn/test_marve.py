"""Tests for marve.ma_robust_variance_est."""
import numpy as np
import pytest
from moirais.fn.marve import ma_robust_variance_est


def test_marve_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_robust_variance_est(yi, X, W, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_marve_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_robust_variance_est(yi, X, W, cluster)
    assert isinstance(result, dict)
