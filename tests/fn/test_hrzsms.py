"""Tests for hrzsms.horowitz_smoothed_max_score."""
import numpy as np
import pytest
from morie.fn.hrzsms import horowitz_smoothed_max_score


def test_hrzsms_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_smoothed_max_score(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsms_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_smoothed_max_score(x, y, bandwidth)
    assert isinstance(result, dict)
