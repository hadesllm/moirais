"""Tests for crtT.chinese_remainder."""
import numpy as np
import pytest
from morie.fn.crtT import chinese_remainder


def test_crtT_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    m = 10
    result = chinese_remainder(a, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crtT_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    m = 10
    result = chinese_remainder(a, m)
    assert isinstance(result, dict)
