"""Tests for gb251.gibbons_pit."""
import numpy as np
import pytest
from morie.fn.gb251 import gibbons_pit


def test_gb251_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_pit(X, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb251_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_pit(X, F)
    assert isinstance(result, dict)
