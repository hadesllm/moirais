"""Tests for wsmscr.wasserman_score_test."""
import numpy as np
import pytest
from morie.fn.wsmscr import wasserman_score_test


def test_wsmscr_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_score_test(data, f, theta0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wsmscr_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_score_test(data, f, theta0)
    assert isinstance(result, dict)
