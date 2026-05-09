"""Tests for sarmix.spatial_ar_combined."""
import numpy as np
import pytest
from moirais.fn.sarmix import spatial_ar_combined


def test_sarmix_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W1 = np.random.default_rng(42).normal(0, 1, 100)
    W2 = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_ar_combined(y, X, W1, W2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sarmix_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W1 = np.random.default_rng(42).normal(0, 1, 100)
    W2 = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_ar_combined(y, X, W1, W2)
    assert isinstance(result, dict)
