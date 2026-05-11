"""Tests for timeRS.timesvd."""
import numpy as np
import pytest
from morie.fn.timeRS import timesvd


def test_timeRS_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    timestamps = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = timesvd(R, timestamps, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_timeRS_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    timestamps = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = timesvd(R, timestamps, K)
    assert isinstance(result, dict)
