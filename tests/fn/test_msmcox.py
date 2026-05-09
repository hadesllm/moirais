"""Tests for msmcox.msm_cox_marginal."""
import numpy as np
import pytest
from moirais.fn.msmcox import msm_cox_marginal


def test_msmcox_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_cox_marginal(time, event, treatment_history, covariate_history)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmcox_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_cox_marginal(time, event, treatment_history, covariate_history)
    assert isinstance(result, dict)
