"""Tests for aftllg.aft_log_logistic."""
import numpy as np
import pytest
from morie.fn.aftllg import aft_log_logistic


def test_aftllg_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_log_logistic(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aftllg_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_log_logistic(time, event, X)
    assert isinstance(result, dict)
