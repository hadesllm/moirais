"""Tests for linucb.linucb."""
import numpy as np
import pytest
from moirais.fn.linucb import linucb


def test_linucb_basic():
    """Test basic functionality."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    arms = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = linucb(context, arms, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linucb_edge():
    """Test edge cases."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    arms = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = linucb(context, arms, alpha)
    assert isinstance(result, dict)
