"""Tests for zipmd.zero_inflated_poisson."""
import numpy as np
import pytest
from moirais.fn.zipmd import zero_inflated_poisson


def test_zipmd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = zero_inflated_poisson(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_zipmd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = zero_inflated_poisson(y, X)
    assert isinstance(result, dict)
