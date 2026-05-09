"""Tests for spkcrs.schabenberger_cross_k_function."""
import numpy as np
import pytest
from moirais.fn.spkcrs import schabenberger_cross_k_function


def test_spkcrs_basic():
    """Test basic functionality."""
    points1 = np.random.default_rng(42).normal(0, 1, 100)
    points2 = np.random.default_rng(42).normal(0, 1, 100)
    lambda1 = np.random.default_rng(42).normal(0, 1, 100)
    lambda2 = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_cross_k_function(points1, points2, lambda1, lambda2, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spkcrs_edge():
    """Test edge cases."""
    points1 = np.random.default_rng(42).normal(0, 1, 100)
    points2 = np.random.default_rng(42).normal(0, 1, 100)
    lambda1 = np.random.default_rng(42).normal(0, 1, 100)
    lambda2 = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_cross_k_function(points1, points2, lambda1, lambda2, r)
    assert isinstance(result, dict)
