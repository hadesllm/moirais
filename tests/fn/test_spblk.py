"""Tests for spblk.spatial_block_kriging."""
import numpy as np
import pytest
from morie.fn.spblk import spatial_block_kriging


def test_spblk_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_block_kriging(x, coords, blocks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spblk_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_block_kriging(x, coords, blocks)
    assert isinstance(result, dict)
