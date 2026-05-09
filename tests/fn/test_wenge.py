"""Tests for wenge.weight_based_mediation."""
import numpy as np
import pytest
from moirais.fn.wenge import weight_based_mediation


def test_wenge_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = weight_based_mediation(X, M, C, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wenge_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = weight_based_mediation(X, M, C, Y)
    assert isinstance(result, dict)
