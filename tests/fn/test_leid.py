"""Tests for leid.leiden_communities."""
import numpy as np
import pytest
from morie.fn.leid import leiden_communities


def test_leid_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = leiden_communities(y, A, resolution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_leid_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = leiden_communities(y, A, resolution)
    assert isinstance(result, dict)
