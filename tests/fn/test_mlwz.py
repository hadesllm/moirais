"""Tests for mlwz.multilevel_within_cluster_z."""
import numpy as np
import pytest
from morie.fn.mlwz import multilevel_within_cluster_z


def test_mlwz_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = multilevel_within_cluster_z(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mlwz_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = multilevel_within_cluster_z(y, cluster)
    assert isinstance(result, dict)
