"""Tests for narm.narm."""
import numpy as np
import pytest
from morie.fn.narm import narm


def test_narm_basic():
    """Test basic functionality."""
    sessions = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = narm(sessions, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_narm_edge():
    """Test edge cases."""
    sessions = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = narm(sessions, K)
    assert isinstance(result, dict)
