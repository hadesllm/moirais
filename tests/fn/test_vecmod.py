"""Tests for vecmod.vector_error_correction."""
import numpy as np
import pytest
from morie.fn.vecmod import vector_error_correction


def test_vecmod_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    r = 10
    lags = 10
    result = vector_error_correction(X, r, lags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vecmod_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    r = 10
    lags = 10
    result = vector_error_correction(X, r, lags)
    assert isinstance(result, dict)
