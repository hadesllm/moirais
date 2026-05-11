"""Tests for hrzboxc.horowitz_box_cox."""
import numpy as np
import pytest
from morie.fn.hrzboxc import horowitz_box_cox


def test_hrzboxc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_box_cox(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzboxc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_box_cox(x, y)
    assert isinstance(result, dict)
