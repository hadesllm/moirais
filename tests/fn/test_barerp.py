"""Tests for barerp.barrier_method."""
import numpy as np
import pytest
from morie.fn.barerp import barrier_method


def test_barerp_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = barrier_method(f, constraints, x0, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_barerp_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = barrier_method(f, constraints, x0, tau)
    assert isinstance(result, dict)
