"""Tests for caustmle.causal_tmle_targeted."""
import numpy as np
import pytest
from moirais.fn.caustmle import causal_tmle_targeted


def test_caustmle_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    Q1 = np.random.default_rng(42).normal(0, 1, 100)
    Q0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_tmle_targeted(y, T, ps, Q1, Q0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_caustmle_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    Q1 = np.random.default_rng(42).normal(0, 1, 100)
    Q0 = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_tmle_targeted(y, T, ps, Q1, Q0)
    assert isinstance(result, dict)
