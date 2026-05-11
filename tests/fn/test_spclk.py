"""Tests for spclk.schabenberger_composite_likelihood."""
import numpy as np
import pytest
from morie.fn.spclk import schabenberger_composite_likelihood


def test_spclk_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_composite_likelihood(coords, z, variogram_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spclk_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_composite_likelihood(coords, z, variogram_model)
    assert isinstance(result, dict)
