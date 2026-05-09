"""Tests for hzbnds.horowitz_manski_bounds."""
import numpy as np
import pytest
from moirais.fn.hzbnds import horowitz_manski_bounds


def test_hzbnds_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = horowitz_manski_bounds(y, R, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hzbnds_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = horowitz_manski_bounds(y, R, y_min, y_max)
    assert isinstance(result, dict)
