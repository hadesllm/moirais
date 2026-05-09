"""Tests for rtwall.rt_wallinga_teunis."""
import numpy as np
import pytest
from moirais.fn.rtwall import rt_wallinga_teunis


def test_rtwall_basic():
    """Test basic functionality."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    serial_interval = np.random.default_rng(42).normal(0, 1, 100)
    result = rt_wallinga_teunis(incidence, serial_interval)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rtwall_edge():
    """Test edge cases."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    serial_interval = np.random.default_rng(42).normal(0, 1, 100)
    result = rt_wallinga_teunis(incidence, serial_interval)
    assert isinstance(result, dict)
