"""Tests for msmaln.aalen_johansen."""
import numpy as np
import pytest
from morie.fn.msmaln import aalen_johansen


def test_msmaln_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    transitions = np.random.default_rng(42).normal(0, 1, 100)
    result = aalen_johansen(time, state, transitions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmaln_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    transitions = np.random.default_rng(42).normal(0, 1, 100)
    result = aalen_johansen(time, state, transitions)
    assert isinstance(result, dict)
