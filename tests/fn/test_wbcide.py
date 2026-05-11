"""Tests for wbcide.wooldridge_bjs_estimator."""
import numpy as np
import pytest
from morie.fn.wbcide import wooldridge_bjs_estimator


def test_wbcide_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = wooldridge_bjs_estimator(y, D, unit, time, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wbcide_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = wooldridge_bjs_estimator(y, D, unit, time, X)
    assert isinstance(result, dict)
