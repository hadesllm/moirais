"""Tests for didfst.did_forest."""
import numpy as np
import pytest
from moirais.fn.didfst import did_forest


def test_didfst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    time = np.linspace(0, 10, 100)
    result = did_forest(y, D, X, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_didfst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    time = np.linspace(0, 10, 100)
    result = did_forest(y, D, X, time)
    assert isinstance(result, dict)
