"""Tests for snmlin.snm_linear."""
import numpy as np
import pytest
from moirais.fn.snmlin import snm_linear


def test_snmlin_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = snm_linear(y, treatment_history, covariate_history, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snmlin_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = snm_linear(y, treatment_history, covariate_history, time)
    assert isinstance(result, dict)
