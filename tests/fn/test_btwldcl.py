"""Tests for btwldcl.boot_wild_cluster."""
import numpy as np
import pytest
from moirais.fn.btwldcl import boot_wild_cluster


def test_btwldcl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_wild_cluster(X, y, cluster, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btwldcl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_wild_cluster(X, y, cluster, B)
    assert isinstance(result, dict)
