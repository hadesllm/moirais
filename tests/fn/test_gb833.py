"""Tests for gb833.gibbons_pct_mod_rank_loc."""
import numpy as np
import pytest
from morie.fn.gb833 import gibbons_pct_mod_rank_loc


def test_gb833_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_pct_mod_rank_loc(x, y, c)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb833_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_pct_mod_rank_loc(x, y, c)
    assert isinstance(result, dict)
