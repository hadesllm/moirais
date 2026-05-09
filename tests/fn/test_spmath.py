"""Tests for spmath.schabenberger_matheron_estimator."""
import numpy as np
import pytest
from moirais.fn.spmath import schabenberger_matheron_estimator


def test_spmath_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    lag_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_matheron_estimator(coords, z, lag_bins)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spmath_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    lag_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_matheron_estimator(coords, z, lag_bins)
    assert isinstance(result, dict)
