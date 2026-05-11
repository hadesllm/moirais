"""Tests for sdurbm.spatial_durbin_model."""
import numpy as np
import pytest
from morie.fn.sdurbm import spatial_durbin_model


def test_sdurbm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_durbin_model(y, X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sdurbm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_durbin_model(y, X, W)
    assert isinstance(result, dict)
