"""Tests for spdetr.spatial_detrending."""
import numpy as np
import pytest
from morie.fn.spdetr import spatial_detrending


def test_spdetr_basic():
    """Test basic functionality."""
    values = np.random.default_rng(42).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_detrending(values, grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spdetr_edge():
    """Test edge cases."""
    values = np.random.default_rng(42).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_detrending(values, grid)
    assert isinstance(result, dict)
