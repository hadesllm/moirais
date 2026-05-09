"""Tests for evstabint.evt_xi_ci_profile."""
import numpy as np
import pytest
from moirais.fn.evstabint import evt_xi_ci_profile


def test_evstabint_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mle = np.random.default_rng(42).normal(0, 1, 100)
    level = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_xi_ci_profile(x, mle, level)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evstabint_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mle = np.random.default_rng(42).normal(0, 1, 100)
    level = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_xi_ci_profile(x, mle, level)
    assert isinstance(result, dict)
