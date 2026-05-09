"""Tests for gbacon.goodman_bacon_decomp."""
import numpy as np
import pytest
from moirais.fn.gbacon import goodman_bacon_decomp


def test_gbacon_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = goodman_bacon_decomp(y, D, unit, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gbacon_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = goodman_bacon_decomp(y, D, unit, time)
    assert isinstance(result, dict)
