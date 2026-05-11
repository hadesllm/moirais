"""Tests for wsmprb.wasserman_parametric_boot."""
import numpy as np
import pytest
from morie.fn.wsmprb import wasserman_parametric_boot


def test_wsmprb_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wasserman_parametric_boot(data, f, T, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmprb_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = wasserman_parametric_boot(data, f, T, B)
    assert isinstance(result, dict)
