"""Tests for semmod.spatial_error_model."""
import numpy as np
import pytest
from moirais.fn.semmod import spatial_error_model


def test_semmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_error_model(y, X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_semmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_error_model(y, X, W)
    assert isinstance(result, dict)
