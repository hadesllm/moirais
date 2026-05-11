"""Tests for msmnbi.msm_negative_binomial."""
import numpy as np
import pytest
from morie.fn.msmnbi import msm_negative_binomial


def test_msmnbi_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = msm_negative_binomial(y, treatment_history, covariate_history, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmnbi_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = msm_negative_binomial(y, treatment_history, covariate_history, alpha)
    assert isinstance(result, dict)
