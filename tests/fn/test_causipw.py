"""Tests for causipw.causal_ipw_truncated."""
import numpy as np
import pytest
from moirais.fn.causipw import causal_ipw_truncated


def test_causipw_basic():
    """Test basic functionality."""
    treat = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = causal_ipw_truncated(treat, y, ps, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causipw_edge():
    """Test edge cases."""
    treat = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = causal_ipw_truncated(treat, y, ps, alpha)
    assert isinstance(result, dict)
