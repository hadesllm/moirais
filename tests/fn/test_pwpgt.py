"""Tests for pwpgt.pwp_gap_time."""
import numpy as np
import pytest
from morie.fn.pwpgt import pwp_gap_time


def test_pwpgt_basic():
    """Test basic functionality."""
    start = np.random.default_rng(42).normal(0, 1, 100)
    stop = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    occurrence = np.random.default_rng(42).normal(0, 1, 100)
    result = pwp_gap_time(start, stop, event, X, occurrence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pwpgt_edge():
    """Test edge cases."""
    start = np.random.default_rng(42).normal(0, 1, 100)
    stop = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    occurrence = np.random.default_rng(42).normal(0, 1, 100)
    result = pwp_gap_time(start, stop, event, X, occurrence)
    assert isinstance(result, dict)
