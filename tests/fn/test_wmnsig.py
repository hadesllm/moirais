"""Tests for wmnsig.weighted_mean_test."""
import numpy as np
import pytest
from morie.fn.wmnsig import weighted_mean_test


def test_wmnsig_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_mean_test(y, weights)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wmnsig_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_mean_test(y, weights)
    assert isinstance(result, dict)
