"""Tests for manski.manski_no_assumption_bounds."""
import numpy as np
import pytest
from morie.fn.manski import manski_no_assumption_bounds


def test_manski_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = manski_no_assumption_bounds(y, D, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_manski_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = manski_no_assumption_bounds(y, D, y_min, y_max)
    assert isinstance(result, dict)
