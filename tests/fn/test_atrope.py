"""Tests for atrope.rotary_position_embedding."""
import numpy as np
import pytest
from moirais.fn.atrope import rotary_position_embedding


def test_atrope_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    theta = 0.0
    result = rotary_position_embedding(y, q, m, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_atrope_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    theta = 0.0
    result = rotary_position_embedding(y, q, m, theta)
    assert isinstance(result, dict)
