"""Tests for volgkr.vol_garman_klass."""
import numpy as np
import pytest
from morie.fn.volgkr import vol_garman_klass


def test_volgkr_basic():
    """Test basic functionality."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    l = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garman_klass(o, h, l, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volgkr_edge():
    """Test edge cases."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    l = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garman_klass(o, h, l, c)
    assert isinstance(result, dict)
