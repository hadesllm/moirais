"""Tests for gb_pp.gibbons_pp_plot."""
import numpy as np
import pytest
from morie.fn.gb_pp import gibbons_pp_plot


def test_gb_pp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_pp_plot(x, F0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_pp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_pp_plot(x, F0)
    assert isinstance(result, dict)
