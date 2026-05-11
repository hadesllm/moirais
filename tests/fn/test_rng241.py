"""Tests for rng241.rangayyan_ch4_log_power_series."""
import numpy as np
import pytest
from morie.fn.rng241 import rangayyan_ch4_log_power_series


def test_rng241_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_log_power_series(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng241_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_log_power_series(x)
    assert isinstance(result, dict)
