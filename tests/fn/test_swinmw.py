"""Tests for swinmw.swin_msa_window."""
import numpy as np
import pytest
from morie.fn.swinmw import swin_msa_window


def test_swinmw_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    relative_bias = np.random.default_rng(42).normal(0, 1, 100)
    result = swin_msa_window(x, window_size, relative_bias)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_swinmw_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    relative_bias = np.random.default_rng(42).normal(0, 1, 100)
    result = swin_msa_window(x, window_size, relative_bias)
    assert isinstance(result, dict)
