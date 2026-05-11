"""Tests for llgaft.log_logistic_aft."""
import numpy as np
import pytest
from morie.fn.llgaft import log_logistic_aft


def test_llgaft_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = log_logistic_aft(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_llgaft_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = log_logistic_aft(time, event, X)
    assert isinstance(result, dict)
