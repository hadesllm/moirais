"""Tests for hmwrst.geron_warm_restarts."""
import numpy as np
import pytest
from morie.fn.hmwrst import geron_warm_restarts


def test_hmwrst_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    T0 = np.random.default_rng(42).normal(0, 1, 100)
    factor = np.random.default_rng(42).normal(0, 1, 100)
    eta_max = 100
    eta_min = 0
    result = geron_warm_restarts(t, T0, factor, eta_max, eta_min)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmwrst_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    T0 = np.random.default_rng(42).normal(0, 1, 100)
    factor = np.random.default_rng(42).normal(0, 1, 100)
    eta_max = 100
    eta_min = 0
    result = geron_warm_restarts(t, T0, factor, eta_max, eta_min)
    assert isinstance(result, dict)
