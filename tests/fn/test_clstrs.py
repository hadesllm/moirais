"""Tests for clstrs.cluster_design."""
import numpy as np
import pytest
from morie.fn.clstrs import cluster_design


def test_clstrs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_design(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clstrs_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_design(y, cluster)
    assert isinstance(result, dict)
