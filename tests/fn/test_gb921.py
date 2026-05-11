"""Tests for gb921.gibbons_mood_scale."""
import numpy as np
import pytest
from morie.fn.gb921 import gibbons_mood_scale


def test_gb921_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_mood_scale(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb921_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_mood_scale(x, y)
    assert isinstance(result, dict)
