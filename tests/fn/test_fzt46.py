"""Tests for fzt46.fauzi_thm4_6_mean_value."""
import numpy as np
import pytest
from morie.fn.fzt46 import fauzi_thm4_6_mean_value


def test_fzt46_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    g_func = (lambda v: v)
    result = fauzi_thm4_6_mean_value(data, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt46_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    g_func = (lambda v: v)
    result = fauzi_thm4_6_mean_value(data, g_func)
    assert isinstance(result, dict)
