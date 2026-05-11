"""Tests for vgrm.variogram."""
import numpy as np
import pytest
from morie.fn.vgrm import variogram


def test_vgrm_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    bins = np.random.default_rng(42).normal(0, 1, 100)
    result = variogram(coords, values, bins)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vgrm_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    bins = np.random.default_rng(42).normal(0, 1, 100)
    result = variogram(coords, values, bins)
    assert isinstance(result, dict)
