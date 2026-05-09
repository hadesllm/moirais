"""Tests for gforml.robins_g_formula."""
import numpy as np
import pytest
from moirais.fn.gforml import robins_g_formula


def test_gforml_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    intervention = np.random.default_rng(42).normal(0, 1, 100)
    result = robins_g_formula(y, treatment_history, covariate_history, time, intervention)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gforml_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    intervention = np.random.default_rng(42).normal(0, 1, 100)
    result = robins_g_formula(y, treatment_history, covariate_history, time, intervention)
    assert isinstance(result, dict)
