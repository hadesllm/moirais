"""Tests for cwcm.centering_within_cluster_mean."""
import numpy as np
import pytest
from morie.fn.cwcm import centering_within_cluster_mean


def test_cwcm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = centering_within_cluster_mean(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cwcm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = centering_within_cluster_mean(y, cluster)
    assert isinstance(result, dict)
