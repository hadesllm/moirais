"""Tests for voltgr.vol_tgarch_fit."""
import numpy as np
import pytest
from moirais.fn.voltgr import vol_tgarch_fit


def test_voltgr_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_tgarch_fit(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_voltgr_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_tgarch_fit(r, init)
    assert isinstance(result, dict)
