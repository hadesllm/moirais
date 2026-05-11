"""Tests for lrtsts.logrank_test."""
import numpy as np
import pytest
from morie.fn.lrtsts import logrank_test


def test_lrtsts_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = logrank_test(time, event, group)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_lrtsts_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = logrank_test(time, event, group)
    assert isinstance(result, dict)
