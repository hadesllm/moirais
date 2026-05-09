"""Tests for grdbs.geron_dbscan_core_point."""
import numpy as np
import pytest
from moirais.fn.grdbs import geron_dbscan_core_point


def test_grdbs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dbscan_core_point(X, eps, min_samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdbs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    eps = np.random.default_rng(42).normal(0, 1, 100)
    min_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dbscan_core_point(X, eps, min_samples)
    assert isinstance(result, dict)
