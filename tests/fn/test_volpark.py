"""Tests for volpark.vol_parkinson_range."""
import numpy as np
import pytest
from moirais.fn.volpark import vol_parkinson_range


def test_volpark_basic():
    """Test basic functionality."""
    high = np.random.default_rng(42).normal(0, 1, 100)
    low = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_parkinson_range(high, low)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volpark_edge():
    """Test edge cases."""
    high = np.random.default_rng(42).normal(0, 1, 100)
    low = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_parkinson_range(high, low)
    assert isinstance(result, dict)
