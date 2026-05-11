"""Tests for survdsc.discrete_time_survival."""
import numpy as np
import pytest
from morie.fn.survdsc import discrete_time_survival


def test_survdsc_basic():
    """Test basic functionality."""
    time_discrete = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = discrete_time_survival(time_discrete, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survdsc_edge():
    """Test edge cases."""
    time_discrete = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = discrete_time_survival(time_discrete, event, X)
    assert isinstance(result, dict)
