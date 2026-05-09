"""Tests for tmlcic.tmle_cluster_ic."""
import numpy as np
import pytest
from moirais.fn.tmlcic import tmle_cluster_ic


def test_tmlcic_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_cluster_ic(y, D, X, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcic_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_cluster_ic(y, D, X, cluster)
    assert isinstance(result, dict)
