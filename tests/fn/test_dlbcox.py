"""Tests for dlbcox.dfbeta_cox."""
import numpy as np
import pytest
from morie.fn.dlbcox import dfbeta_cox


def test_dlbcox_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dfbeta_cox(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dlbcox_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dfbeta_cox(time, event, X)
    assert isinstance(result, dict)
