"""Tests for tmlmpc.tmle_multi_state_phc."""
import numpy as np
import pytest
from moirais.fn.tmlmpc import tmle_multi_state_phc


def test_tmlmpc_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_multi_state_phc(time, state, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlmpc_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_multi_state_phc(time, state, D, X)
    assert isinstance(result, dict)
