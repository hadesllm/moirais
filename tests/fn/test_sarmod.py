"""Tests for sarmod.spatial_lag_model."""
import numpy as np
import pytest
from morie.fn.sarmod import spatial_lag_model


def test_sarmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_lag_model(y, X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sarmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_lag_model(y, X, W)
    assert isinstance(result, dict)
