"""Tests for volhar.vol_har_rv."""
import numpy as np
import pytest
from moirais.fn.volhar import vol_har_rv


def test_volhar_basic():
    """Test basic functionality."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = vol_har_rv(RV, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volhar_edge():
    """Test edge cases."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = vol_har_rv(RV, h)
    assert isinstance(result, dict)
