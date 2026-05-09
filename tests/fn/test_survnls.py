"""Tests for survnls.nonlinear_least_squares_surv."""
import numpy as np
import pytest
from moirais.fn.survnls import nonlinear_least_squares_surv


def test_survnls_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = nonlinear_least_squares_surv(time, event, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survnls_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = nonlinear_least_squares_surv(time, event, model)
    assert isinstance(result, dict)
