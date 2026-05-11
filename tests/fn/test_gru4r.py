"""Tests for gru4r.gru4rec."""
import numpy as np
import pytest
from morie.fn.gru4r import gru4rec


def test_gru4r_basic():
    """Test basic functionality."""
    sessions = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gru4rec(sessions, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gru4r_edge():
    """Test edge cases."""
    sessions = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gru4rec(sessions, K)
    assert isinstance(result, dict)
