"""Tests for baysrnd.shrinkage_random."""
import numpy as np
import pytest
from morie.fn.baysrnd import shrinkage_random


def test_baysrnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_random(y, X, group)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baysrnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_random(y, X, group)
    assert isinstance(result, dict)
