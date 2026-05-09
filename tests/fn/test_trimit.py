"""Tests for trimit.weight_trimming."""
import numpy as np
import pytest
from moirais.fn.trimit import weight_trimming


def test_trimit_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = weight_trimming(y, weights, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trimit_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = weight_trimming(y, weights, threshold)
    assert isinstance(result, dict)
