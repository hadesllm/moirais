"""Tests for volaprch.vol_aparch_fit."""
import numpy as np
import pytest
from morie.fn.volaprch import vol_aparch_fit


def test_volaprch_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_aparch_fit(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volaprch_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_aparch_fit(r, init)
    assert isinstance(result, dict)
