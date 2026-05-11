"""Tests for bndmsg.bound_missing_outcome."""
import numpy as np
import pytest
from morie.fn.bndmsg import bound_missing_outcome


def test_bndmsg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = bound_missing_outcome(y, R, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndmsg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = bound_missing_outcome(y, R, y_min, y_max)
    assert isinstance(result, dict)
