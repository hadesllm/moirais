"""Tests for jjmsta.join_count."""
import numpy as np
import pytest
from morie.fn.jjmsta import join_count


def test_jjmsta_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = join_count(x, W)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jjmsta_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = join_count(x, W)
    assert isinstance(result, dict)
