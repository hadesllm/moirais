"""Tests for grql.geron_q_learning_update."""
import numpy as np
import pytest
from moirais.fn.grql import geron_q_learning_update


def test_grql_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    a = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_q_learning_update(Q, s, a, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grql_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    a = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_q_learning_update(Q, s, a, r, s_next, alpha, gamma)
    assert isinstance(result, dict)
