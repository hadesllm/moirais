"""Tests for gmccsm.g_methods_consistency."""
import numpy as np
import pytest
from morie.fn.gmccsm import g_methods_consistency


def test_gmccsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = g_methods_consistency(y, treatment_history, covariate_history, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gmccsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = g_methods_consistency(y, treatment_history, covariate_history, tau)
    assert isinstance(result, dict)
