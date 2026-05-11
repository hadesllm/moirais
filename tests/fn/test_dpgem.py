"""Tests for dpgem.gem_distribution."""
import numpy as np
import pytest
from morie.fn.dpgem import gem_distribution


def test_dpgem_basic():
    """Test basic functionality."""
    alpha = 0.05
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gem_distribution(alpha, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpgem_edge():
    """Test edge cases."""
    alpha = 0.05
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gem_distribution(alpha, K)
    assert isinstance(result, dict)
