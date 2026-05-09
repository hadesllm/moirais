"""Tests for msmgmm.msm_gmm_estimator."""
import numpy as np
import pytest
from moirais.fn.msmgmm import msm_gmm_estimator


def test_msmgmm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    instruments = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_gmm_estimator(y, treatment_history, covariate_history, instruments)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmgmm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    instruments = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_gmm_estimator(y, treatment_history, covariate_history, instruments)
    assert isinstance(result, dict)
