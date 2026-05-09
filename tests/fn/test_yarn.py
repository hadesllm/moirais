"""Tests for yarn.yarn_context_scaling."""
import numpy as np
import pytest
from moirais.fn.yarn import yarn_context_scaling


def test_yarn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    theta = 0.0
    s = 90
    beta_fast = np.random.default_rng(42).normal(0, 1, 100)
    beta_slow = np.random.default_rng(42).normal(0, 1, 100)
    result = yarn_context_scaling(y, q, m, theta, s, beta_fast, beta_slow)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_yarn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    theta = 0.0
    s = 90
    beta_fast = np.random.default_rng(42).normal(0, 1, 100)
    beta_slow = np.random.default_rng(42).normal(0, 1, 100)
    result = yarn_context_scaling(y, q, m, theta, s, beta_fast, beta_slow)
    assert isinstance(result, dict)
