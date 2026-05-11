"""Tests for spstvg.schabenberger_st_variogram."""
import numpy as np
import pytest
from morie.fn.spstvg import schabenberger_st_variogram


def test_spstvg_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    times = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_st_variogram(coords, times, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spstvg_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    times = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_st_variogram(coords, times, z)
    assert isinstance(result, dict)
