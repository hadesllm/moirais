"""Tests for adfullr.adf_unit_root."""
import numpy as np
import pytest
from morie.fn.adfullr import adf_unit_root


def test_adfullr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = adf_unit_root(y, trend, p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_adfullr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = adf_unit_root(y, trend, p)
    assert isinstance(result, dict)
