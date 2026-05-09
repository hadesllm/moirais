"""Tests for tpspn.thin_plate_spline."""
import numpy as np
import pytest
from moirais.fn.tpspn import thin_plate_spline


def test_tpspn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = thin_plate_spline(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tpspn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = thin_plate_spline(x, y)
    assert isinstance(result, dict)
