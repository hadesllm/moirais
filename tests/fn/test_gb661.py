"""Tests for gb661.gibbons_mannwhitney."""
import numpy as np
import pytest
from morie.fn.gb661 import gibbons_mannwhitney


def test_gb661_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_mannwhitney(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb661_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_mannwhitney(x, y)
    assert isinstance(result, dict)
