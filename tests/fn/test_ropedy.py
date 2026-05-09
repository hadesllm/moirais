"""Tests for ropedy.rope_ntk_dynamic."""
import numpy as np
import pytest
from moirais.fn.ropedy import rope_ntk_dynamic


def test_ropedy_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    theta = 0.0
    L_new = np.random.default_rng(42).normal(0, 1, 100)
    L_train = np.random.default_rng(42).normal(0, 1, 100)
    result = rope_ntk_dynamic(y, q, m, theta, L_new, L_train)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ropedy_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    theta = 0.0
    L_new = np.random.default_rng(42).normal(0, 1, 100)
    L_train = np.random.default_rng(42).normal(0, 1, 100)
    result = rope_ntk_dynamic(y, q, m, theta, L_new, L_train)
    assert isinstance(result, dict)
