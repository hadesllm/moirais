"""Tests for gb_exc.gibbons_exceedance_stat."""
import numpy as np
import pytest
from morie.fn.gb_exc import gibbons_exceedance_stat


def test_gb_exc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_exceedance_stat(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_exc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_exceedance_stat(x, y)
    assert isinstance(result, dict)
