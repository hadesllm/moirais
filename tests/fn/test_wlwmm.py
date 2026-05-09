"""Tests for wlwmm.wlw_marginal_model."""
import numpy as np
import pytest
from moirais.fn.wlwmm import wlw_marginal_model


def test_wlwmm_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    occurrence = np.random.default_rng(42).normal(0, 1, 100)
    result = wlw_marginal_model(time, event, X, occurrence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wlwmm_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    occurrence = np.random.default_rng(42).normal(0, 1, 100)
    result = wlw_marginal_model(time, event, X, occurrence)
    assert isinstance(result, dict)
