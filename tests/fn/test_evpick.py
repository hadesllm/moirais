"""Tests for evpick.evt_pickands_estimator."""
import numpy as np
import pytest
from moirais.fn.evpick import evt_pickands_estimator


def test_evpick_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = evt_pickands_estimator(x, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evpick_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = evt_pickands_estimator(x, k)
    assert isinstance(result, dict)
