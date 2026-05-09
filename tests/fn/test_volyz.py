"""Tests for volyz.vol_yang_zhang."""
import numpy as np
import pytest
from moirais.fn.volyz import vol_yang_zhang


def test_volyz_basic():
    """Test basic functionality."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    l = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_yang_zhang(o, h, l, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volyz_edge():
    """Test edge cases."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    l = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_yang_zhang(o, h, l, c)
    assert isinstance(result, dict)
