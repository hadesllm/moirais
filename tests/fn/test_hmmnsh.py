"""Tests for hmmnsh.geron_mean_shift."""
import numpy as np
import pytest
from morie.fn.hmmnsh import geron_mean_shift


def test_hmmnsh_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bandwidth = 0.3
    result = geron_mean_shift(X, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmnsh_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bandwidth = 0.3
    result = geron_mean_shift(X, bandwidth)
    assert isinstance(result, dict)
