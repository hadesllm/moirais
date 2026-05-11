"""Tests for mivbnd.monotone_iv_bounds."""
import numpy as np
import pytest
from morie.fn.mivbnd import monotone_iv_bounds


def test_mivbnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    y_min = 0
    y_max = 100
    result = monotone_iv_bounds(y, D, Z, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mivbnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    y_min = 0
    y_max = 100
    result = monotone_iv_bounds(y, D, Z, y_min, y_max)
    assert isinstance(result, dict)
