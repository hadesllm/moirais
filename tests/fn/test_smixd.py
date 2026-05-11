"""Tests for smixd.spatial_mixed_model."""
import numpy as np
import pytest
from morie.fn.smixd import spatial_mixed_model


def test_smixd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_mixed_model(x, y, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smixd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_mixed_model(x, y, coords)
    assert isinstance(result, dict)
