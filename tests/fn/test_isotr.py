"""Tests for isotr.isotonic_regression_disparity."""
import numpy as np
import pytest
from morie.fn.isotr import isotonic_regression_disparity


def test_isotr_basic():
    """Test basic functionality."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    delta_rank = np.random.default_rng(42).normal(0, 1, 100)
    result = isotonic_regression_disparity(D, delta_rank)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_isotr_edge():
    """Test edge cases."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    delta_rank = np.random.default_rng(42).normal(0, 1, 100)
    result = isotonic_regression_disparity(D, delta_rank)
    assert isinstance(result, dict)
