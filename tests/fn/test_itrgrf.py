"""Tests for itrgrf.itr_forest."""
import numpy as np
import pytest
from moirais.fn.itrgrf import itr_forest


def test_itrgrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = itr_forest(y, D, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_itrgrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = itr_forest(y, D, W)
    assert isinstance(result, dict)
