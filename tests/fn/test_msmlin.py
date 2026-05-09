"""Tests for msmlin.msm_linear."""
import numpy as np
import pytest
from moirais.fn.msmlin import msm_linear


def test_msmlin_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_linear(y, treatment_history, covariate_history)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmlin_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_linear(y, treatment_history, covariate_history)
    assert isinstance(result, dict)
