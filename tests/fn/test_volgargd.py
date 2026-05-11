"""Tests for volgargd.vol_garch_ged."""
import numpy as np
import pytest
from morie.fn.volgargd import vol_garch_ged


def test_volgargd_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_ged(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volgargd_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch_ged(r, init)
    assert isinstance(result, dict)
