"""Tests for volgargt.vol_garch_t."""
import numpy as np
import pytest
from morie.fn.volgargt import vol_garch_t


def test_volgargt_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_t(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volgargt_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_t(r, init)
    assert isinstance(result, dict)
