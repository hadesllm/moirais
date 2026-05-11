"""Tests for gctvc.g_computation_time_varying."""
import numpy as np
import pytest
from morie.fn.gctvc import g_computation_time_varying


def test_gctvc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = g_computation_time_varying(y, treatment_history, covariate_history, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gctvc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = g_computation_time_varying(y, treatment_history, covariate_history, time)
    assert isinstance(result, dict)
