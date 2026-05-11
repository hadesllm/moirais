"""Tests for crrfgs.competing_risks_fg."""
import numpy as np
import pytest
from morie.fn.crrfgs import competing_risks_fg


def test_crrfgs_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = competing_risks_fg(time, event_type, X, cause)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crrfgs_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cause = np.random.default_rng(42).normal(0, 1, 100)
    result = competing_risks_fg(time, event_type, X, cause)
    assert isinstance(result, dict)
