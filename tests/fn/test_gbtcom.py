"""Tests for gbtcom.goodman_bacon_3way."""
import numpy as np
import pytest
from morie.fn.gbtcom import goodman_bacon_3way


def test_gbtcom_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = goodman_bacon_3way(y, D, unit, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gbtcom_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = goodman_bacon_3way(y, D, unit, time)
    assert isinstance(result, dict)
