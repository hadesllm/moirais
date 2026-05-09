"""Tests for dyntmt.dynamic_marginal_msm."""
import numpy as np
import pytest
from moirais.fn.dyntmt import dynamic_marginal_msm


def test_dyntmt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_history = np.random.default_rng(42).normal(0, 1, 100)
    H_history = np.random.default_rng(42).normal(0, 1, 100)
    regime_fn = (lambda v: v)
    result = dynamic_marginal_msm(y, D_history, H_history, regime_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dyntmt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_history = np.random.default_rng(42).normal(0, 1, 100)
    H_history = np.random.default_rng(42).normal(0, 1, 100)
    regime_fn = (lambda v: v)
    result = dynamic_marginal_msm(y, D_history, H_history, regime_fn)
    assert isinstance(result, dict)
