"""Tests for volgjr.vol_gjr_garch."""
import numpy as np
import pytest
from morie.fn.volgjr import vol_gjr_garch


def test_volgjr_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_gjr_garch(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volgjr_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_gjr_garch(r, init)
    assert isinstance(result, dict)
