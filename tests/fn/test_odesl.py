"""Tests for odesl.ode_symbolic."""
import numpy as np
import pytest
from moirais.fn.odesl import ode_symbolic


def test_odesl_basic():
    """Test basic functionality."""
    ode = np.random.default_rng(42).normal(0, 1, 100)
    result = ode_symbolic(ode)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_odesl_edge():
    """Test edge cases."""
    ode = np.random.default_rng(42).normal(0, 1, 100)
    result = ode_symbolic(ode)
    assert isinstance(result, dict)
