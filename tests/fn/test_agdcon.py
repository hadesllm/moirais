"""Tests for agdcon.agd_constrained."""
import numpy as np
import pytest
from morie.fn.agdcon import agd_constrained


def test_agdcon_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    project = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = agd_constrained(f, grad_f, project, x0, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agdcon_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    project = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = agd_constrained(f, grad_f, project, x0, steps)
    assert isinstance(result, dict)
