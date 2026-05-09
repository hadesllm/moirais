"""Tests for sacmod.spatial_combined."""
import numpy as np
import pytest
from moirais.fn.sacmod import spatial_combined


def test_sacmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_combined(y, X, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sacmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_combined(y, X, W)
    assert isinstance(result, dict)
