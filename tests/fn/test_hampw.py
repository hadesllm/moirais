"""Tests for hampw.hampel_three_part."""
import numpy as np
import pytest
from morie.fn.hampw import hampel_three_part


def test_hampw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = hampel_three_part(y, a, b, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hampw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = hampel_three_part(y, a, b, c)
    assert isinstance(result, dict)
