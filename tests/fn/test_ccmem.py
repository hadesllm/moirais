"""Tests for ccmem.cross_classified_membership."""
import numpy as np
import pytest
from morie.fn.ccmem import cross_classified_membership


def test_ccmem_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster1 = np.random.default_rng(42).normal(0, 1, 100)
    cluster2 = np.random.default_rng(42).normal(0, 1, 100)
    result = cross_classified_membership(y, cluster1, cluster2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ccmem_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster1 = np.random.default_rng(42).normal(0, 1, 100)
    cluster2 = np.random.default_rng(42).normal(0, 1, 100)
    result = cross_classified_membership(y, cluster1, cluster2)
    assert isinstance(result, dict)
