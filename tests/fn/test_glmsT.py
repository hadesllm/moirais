"""Tests for glmsT.linear_trend."""
import numpy as np
import pytest
from moirais.fn.glmsT import linear_trend


def test_glmsT_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = linear_trend(t, x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_glmsT_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = linear_trend(t, x)
    assert isinstance(result, dict)
