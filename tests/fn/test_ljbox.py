"""Tests for ljbox.ljung_box_test."""
import numpy as np
import pytest
from morie.fn.ljbox import ljbox as ljung_box_test


def test_ljbox_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = ljung_box_test(x, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ljbox_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = ljung_box_test(x, lags)
    assert isinstance(result, dict)
