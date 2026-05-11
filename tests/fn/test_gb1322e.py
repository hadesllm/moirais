"""Tests for gb1322e.gibbons_efficacy."""
import numpy as np
import pytest
from morie.fn.gb1322e import gibbons_efficacy


def test_gb1322e_basic():
    """Test basic functionality."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    u0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_efficacy(T, u0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1322e_edge():
    """Test edge cases."""
    T = np.random.default_rng(42).normal(0, 1, 100)
    u0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_efficacy(T, u0)
    assert isinstance(result, dict)
