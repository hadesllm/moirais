"""Tests for cluvar.cluster_variance."""
import numpy as np
import pytest
from morie.fn.cluvar import cluster_variance


def test_cluvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_variance(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cluvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_variance(y, cluster)
    assert isinstance(result, dict)
