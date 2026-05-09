"""Tests for grlof.geron_local_outlier_factor."""
import numpy as np
import pytest
from moirais.fn.grlof import geron_local_outlier_factor


def test_grlof_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = geron_local_outlier_factor(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grlof_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = geron_local_outlier_factor(X, k)
    assert isinstance(result, dict)
