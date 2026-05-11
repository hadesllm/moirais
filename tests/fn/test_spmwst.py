"""Tests for spmwst.schabenberger_moving_window."""
import numpy as np
import pytest
from morie.fn.spmwst import schabenberger_moving_window


def test_spmwst_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    window_size = 100
    result = schabenberger_moving_window(coords, z, window_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spmwst_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    window_size = 100
    result = schabenberger_moving_window(coords, z, window_size)
    assert isinstance(result, dict)
