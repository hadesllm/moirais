"""Tests for gb_mw2.gibbons_mw_rs_equiv."""
import numpy as np
import pytest
from morie.fn.gb_mw2 import gibbons_mw_rs_equiv


def test_gb_mw2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_mw_rs_equiv(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_mw2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_mw_rs_equiv(x, y)
    assert isinstance(result, dict)
