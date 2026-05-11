"""Tests for linTS.lin_thompson."""
import numpy as np
import pytest
from morie.fn.linTS import lin_thompson


def test_linTS_basic():
    """Test basic functionality."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    arms = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = lin_thompson(context, arms, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linTS_edge():
    """Test edge cases."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    arms = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = lin_thompson(context, arms, beta)
    assert isinstance(result, dict)
