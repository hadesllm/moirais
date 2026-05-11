"""Tests for snaivf.seasonal_naive."""
import numpy as np
import pytest
from morie.fn.snaivf import seasonal_naive


def test_snaivf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    m = 10
    h = 0.3
    result = seasonal_naive(y, m, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snaivf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    m = 10
    h = 0.3
    result = seasonal_naive(y, m, h)
    assert isinstance(result, dict)
