"""Tests for detfdt.detrended_fluctuation."""
import numpy as np
import pytest
from morie.fn.detfdt import detrended_fluctuation


def test_detfdt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = detrended_fluctuation(y, scales)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_detfdt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    scales = np.random.default_rng(42).normal(0, 1, 100)
    result = detrended_fluctuation(y, scales)
    assert isinstance(result, dict)
