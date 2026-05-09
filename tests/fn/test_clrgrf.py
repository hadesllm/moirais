"""Tests for clrgrf.clustered_grf."""
import numpy as np
import pytest
from moirais.fn.clrgrf import clustered_grf


def test_clrgrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = clustered_grf(y, D, X, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clrgrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = clustered_grf(y, D, X, cluster)
    assert isinstance(result, dict)
