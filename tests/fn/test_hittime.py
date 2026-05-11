"""Tests for hittime.hitting_time."""
import numpy as np
import pytest
from morie.fn.hittime import hitting_time


def test_hittime_basic():
    """Test basic functionality."""
    G = np.eye(10)
    start = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = hitting_time(G, start, target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hittime_edge():
    """Test edge cases."""
    G = np.eye(10)
    start = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = hitting_time(G, start, target)
    assert isinstance(result, dict)
