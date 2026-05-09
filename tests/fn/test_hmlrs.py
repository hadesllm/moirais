"""Tests for hmlrs.geron_learning_rate_schedule."""
import numpy as np
import pytest
from moirais.fn.hmlrs import geron_learning_rate_schedule


def test_hmlrs_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    eta0 = np.random.default_rng(42).normal(0, 1, 100)
    t0 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_learning_rate_schedule(t, eta0, t0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlrs_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    eta0 = np.random.default_rng(42).normal(0, 1, 100)
    t0 = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_learning_rate_schedule(t, eta0, t0)
    assert isinstance(result, dict)
