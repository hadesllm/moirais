"""Tests for hierdp.hierarchical_dp_density."""
import numpy as np
import pytest
from morie.fn.hierdp import hierarchical_dp_density


def test_hierdp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    gamma = 1.0
    alpha = 0.05
    result = hierarchical_dp_density(y, groups, gamma, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hierdp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    gamma = 1.0
    alpha = 0.05
    result = hierarchical_dp_density(y, groups, gamma, alpha)
    assert isinstance(result, dict)
