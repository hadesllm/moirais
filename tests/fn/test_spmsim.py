"""Tests for spmsim.schabenberger_mgwr_bandwidth."""
import numpy as np
import pytest
from morie.fn.spmsim import schabenberger_mgwr_bandwidth


def test_spmsim_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = schabenberger_mgwr_bandwidth(x, y, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spmsim_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = schabenberger_mgwr_bandwidth(x, y, coords)
    assert isinstance(result, dict)
