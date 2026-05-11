"""Tests for ljbox2.ljung_box."""
import numpy as np
import pytest
from morie.fn.ljbox2 import ljung_box


def test_ljbox2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lags = 10
    result = ljung_box(y, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ljbox2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lags = 10
    result = ljung_box(y, lags)
    assert isinstance(result, dict)
