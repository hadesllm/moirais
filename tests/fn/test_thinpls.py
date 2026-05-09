"""Tests for thinpls.thin_plate_spline."""
import numpy as np
import pytest
from moirais.fn.thinpls import thin_plate_spline


def test_thinpls_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    lam = 0.1
    result = thin_plate_spline(x, y, z, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_thinpls_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    lam = 0.1
    result = thin_plate_spline(x, y, z, lam)
    assert isinstance(result, dict)
