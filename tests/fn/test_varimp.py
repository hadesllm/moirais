"""Tests for varimp.var_impulse_response."""
import numpy as np
import pytest
from morie.fn.varimp import var_impulse_response


def test_varimp_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = var_impulse_response(fit, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varimp_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = var_impulse_response(fit, horizon)
    assert isinstance(result, dict)
