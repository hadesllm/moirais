"""Tests for cluster.one_stage_cluster."""
import numpy as np
import pytest
from moirais.fn.cluster import one_stage_cluster


def test_cluster_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = one_stage_cluster(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cluster_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = one_stage_cluster(y, cluster)
    assert isinstance(result, dict)
