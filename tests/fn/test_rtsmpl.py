"""Tests for rtsmpl.rt_serial_interval."""
import numpy as np
import pytest
from morie.fn.rtsmpl import rt_serial_interval


def test_rtsmpl_basic():
    """Test basic functionality."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    serial_interval = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rt_serial_interval(incidence, serial_interval, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rtsmpl_edge():
    """Test edge cases."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    serial_interval = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rt_serial_interval(incidence, serial_interval, window)
    assert isinstance(result, dict)
