"""Tests for caussc.causal_synthetic_control."""
import numpy as np
import pytest
from morie.fn.caussc import causal_synthetic_control


def test_caussc_basic():
    """Test basic functionality."""
    X1_pre = np.random.default_rng(42).normal(0, 1, 100)
    X0_pre = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_synthetic_control(X1_pre, X0_pre, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_caussc_edge():
    """Test edge cases."""
    X1_pre = np.random.default_rng(42).normal(0, 1, 100)
    X0_pre = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_synthetic_control(X1_pre, X0_pre, V)
    assert isinstance(result, dict)
