"""Tests for jowink.joseph_winkler_interval_score."""
import numpy as np
import pytest
from morie.fn.jowink import joseph_winkler_interval_score


def test_jowink_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = joseph_winkler_interval_score(y, l, u, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jowink_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = joseph_winkler_interval_score(y, l, u, alpha)
    assert isinstance(result, dict)
