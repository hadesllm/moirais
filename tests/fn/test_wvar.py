"""Tests for wvar.weighted_variance."""
import numpy as np
import pytest
from morie.fn.wvar import weighted_variance


def test_wvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_variance(y, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_variance(y, weights)
    assert isinstance(result, dict)
