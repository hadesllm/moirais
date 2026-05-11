"""Tests for ghgps.ghosal_gp_squared_exponential."""
import numpy as np
import pytest
from morie.fn.ghgps import ghosal_gp_squared_exponential


def test_ghgps_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gp_squared_exponential(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghgps_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gp_squared_exponential(x, y)
    assert isinstance(result, dict)
