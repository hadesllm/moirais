"""Tests for sppow.schabenberger_power_variogram."""
import numpy as np
import pytest
from moirais.fn.sppow import schabenberger_power_variogram


def test_sppow_basic():
    """Test basic functionality."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = schabenberger_power_variogram(h, nugget, c1, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sppow_edge():
    """Test edge cases."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = schabenberger_power_variogram(h, nugget, c1, alpha)
    assert isinstance(result, dict)
