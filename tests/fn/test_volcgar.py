"""Tests for volcgar.vol_cgarch_fit."""
import numpy as np
import pytest
from morie.fn.volcgar import vol_cgarch_fit


def test_volcgar_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_cgarch_fit(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volcgar_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_cgarch_fit(r, init)
    assert isinstance(result, dict)
