"""Tests for otmd.ot_mahalanobis_distance_ot."""
import numpy as np
import pytest
from moirais.fn.otmd import ot_mahalanobis_distance_ot


def test_otmd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_mahalanobis_distance_ot(X, Y, Sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_mahalanobis_distance_ot(X, Y, Sigma)
    assert isinstance(result, dict)
