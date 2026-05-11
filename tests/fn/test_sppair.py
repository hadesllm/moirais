"""Tests for sppair.schabenberger_pair_correlation."""
import numpy as np
import pytest
from morie.fn.sppair import schabenberger_pair_correlation


def test_sppair_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    lambda_est = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_pair_correlation(points, lambda_est, r)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_sppair_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    lambda_est = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_pair_correlation(points, lambda_est, r)
    assert isinstance(result, dict)
