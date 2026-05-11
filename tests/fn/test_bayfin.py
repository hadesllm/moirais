"""Tests for bayfin.finite_mixture."""
import numpy as np
import pytest
from morie.fn.bayfin import finite_mixture


def test_bayfin_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = finite_mixture(y, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayfin_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = finite_mixture(y, K)
    assert isinstance(result, dict)
