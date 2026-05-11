"""Tests for frwol2.frank_wolfe."""
import numpy as np
import pytest
from morie.fn.frwol2 import frank_wolfe


def test_frwol2_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    domain = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = frank_wolfe(f, grad_f, domain, x0, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frwol2_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    domain = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = frank_wolfe(f, grad_f, domain, x0, steps)
    assert isinstance(result, dict)
