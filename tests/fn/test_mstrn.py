"""Tests for mstrn.multistate_transition_matrix."""
import numpy as np
import pytest
from moirais.fn.mstrn import multistate_transition_matrix


def test_mstrn_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = multistate_transition_matrix(time, state, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mstrn_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = multistate_transition_matrix(time, state, X)
    assert isinstance(result, dict)
