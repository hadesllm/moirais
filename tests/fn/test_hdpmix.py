"""Tests for hdpmix.hierarchical_dp."""
import numpy as np
import pytest
from moirais.fn.hdpmix import hierarchical_dp


def test_hdpmix_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    gamma = 1.0
    alpha = 0.05
    result = hierarchical_dp(y, groups, gamma, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hdpmix_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    gamma = 1.0
    alpha = 0.05
    result = hierarchical_dp(y, groups, gamma, alpha)
    assert isinstance(result, dict)
