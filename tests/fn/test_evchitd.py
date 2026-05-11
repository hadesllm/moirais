"""Tests for evchitd.evt_chi_tail_dependence."""
import numpy as np
import pytest
from morie.fn.evchitd import evt_chi_tail_dependence


def test_evchitd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_chi_tail_dependence(x, y, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evchitd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_chi_tail_dependence(x, y, u)
    assert isinstance(result, dict)
