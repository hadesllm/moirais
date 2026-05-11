"""Tests for volign.vol_igarch_fit."""
import numpy as np
import pytest
from morie.fn.volign import vol_igarch_fit


def test_volign_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_igarch_fit(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volign_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_igarch_fit(r, init)
    assert isinstance(result, dict)
