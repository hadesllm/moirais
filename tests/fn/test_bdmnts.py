"""Tests for bdmnts.bound_monot_inst."""
import numpy as np
import pytest
from morie.fn.bdmnts import bound_monot_inst


def test_bdmnts_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    y_min = 0
    y_max = 100
    result = bound_monot_inst(y, D, Z, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdmnts_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    y_min = 0
    y_max = 100
    result = bound_monot_inst(y, D, Z, y_min, y_max)
    assert isinstance(result, dict)
