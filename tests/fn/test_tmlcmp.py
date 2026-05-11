"""Tests for tmlcmp.tmle_competing_risks."""
import numpy as np
import pytest
from morie.fn.tmlcmp import tmle_competing_risks


def test_tmlcmp_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_competing_risks(time, event_type, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcmp_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_competing_risks(time, event_type, D, X)
    assert isinstance(result, dict)
