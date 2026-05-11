"""Tests for cookd.cooks_distance."""
import numpy as np
import pytest
from morie.fn.cookd import cooks_distance


def test_cookd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = cooks_distance(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cookd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = cooks_distance(y, X)
    assert isinstance(result, dict)
