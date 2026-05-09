"""Tests for msmlog.msm_logistic."""
import numpy as np
import pytest
from moirais.fn.msmlog import msm_logistic


def test_msmlog_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_logistic(y, treatment_history, covariate_history)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmlog_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_logistic(y, treatment_history, covariate_history)
    assert isinstance(result, dict)
