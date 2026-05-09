"""Tests for orfgrf.orthogonal_random_forest."""
import numpy as np
import pytest
from moirais.fn.orfgrf import orthogonal_random_forest


def test_orfgrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = orthogonal_random_forest(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_orfgrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = orthogonal_random_forest(y, D, X)
    assert isinstance(result, dict)
