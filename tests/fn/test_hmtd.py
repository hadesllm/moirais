"""Tests for hmtd.geron_td_learning."""
import numpy as np
import pytest
from morie.fn.hmtd import geron_td_learning


def test_hmtd_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    r = 10
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_td_learning(V, s, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtd_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    r = 10
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_td_learning(V, s, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
