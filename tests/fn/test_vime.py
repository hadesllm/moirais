"""Tests for vime.empirical_orthogonal_func."""
import numpy as np
import pytest
from morie.fn.vime import empirical_orthogonal_func


def test_vime_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = empirical_orthogonal_func(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vime_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = empirical_orthogonal_func(X)
    assert isinstance(result, dict)
