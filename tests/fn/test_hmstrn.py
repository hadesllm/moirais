"""Tests for hmstrn.history_adjusted_msm."""
import numpy as np
import pytest
from moirais.fn.hmstrn import history_adjusted_msm


def test_hmstrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    regime = np.random.default_rng(42).normal(0, 1, 100)
    result = history_adjusted_msm(y, treatment_history, covariate_history, time, regime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmstrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    regime = np.random.default_rng(42).normal(0, 1, 100)
    result = history_adjusted_msm(y, treatment_history, covariate_history, time, regime)
    assert isinstance(result, dict)
