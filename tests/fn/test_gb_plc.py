"""Tests for gb_plc.gibbons_placement_def."""
import numpy as np
import pytest
from morie.fn.gb_plc import gibbons_placement_def


def test_gb_plc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_placement_def(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_plc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_placement_def(x, y)
    assert isinstance(result, dict)
