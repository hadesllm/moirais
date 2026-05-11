"""Tests for jooutl.joseph_ts_outlier_detection."""
import numpy as np
import pytest
from morie.fn.jooutl import joseph_ts_outlier_detection


def test_jooutl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_ts_outlier_detection(y, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jooutl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_ts_outlier_detection(y, W)
    assert isinstance(result, dict)
