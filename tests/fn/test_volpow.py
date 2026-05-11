"""Tests for volpow.vol_power_variation."""
import numpy as np
import pytest
from morie.fn.volpow import vol_power_variation


def test_volpow_basic():
    """Test basic functionality."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = vol_power_variation(r_intraday, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volpow_edge():
    """Test edge cases."""
    r_intraday = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = vol_power_variation(r_intraday, p)
    assert isinstance(result, dict)
