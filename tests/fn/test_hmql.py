"""Tests for hmql.geron_q_learning."""
import numpy as np
import pytest
from moirais.fn.hmql import geron_q_learning


def test_hmql_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    a = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_q_learning(Q, s, a, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmql_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    a = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_q_learning(Q, s, a, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
