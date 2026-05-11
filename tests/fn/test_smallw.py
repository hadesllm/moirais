"""Tests for smallw.small_world_sigma."""
import numpy as np
import pytest
from morie.fn.smallw import small_world_sigma


def test_smallw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = small_world_sigma(y, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smallw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = small_world_sigma(y, A)
    assert isinstance(result, dict)
