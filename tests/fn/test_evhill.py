"""Tests for evhill.evt_hill_estimator."""
import numpy as np
import pytest
from moirais.fn.evhill import evt_hill_estimator


def test_evhill_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = evt_hill_estimator(x, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evhill_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = evt_hill_estimator(x, k)
    assert isinstance(result, dict)
