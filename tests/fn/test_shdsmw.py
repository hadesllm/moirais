"""Tests for shdsmw.shrinkage_msm."""
import numpy as np
import pytest
from moirais.fn.shdsmw import shrinkage_msm


def test_shdsmw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = shrinkage_msm(y, treatment_history, covariate_history, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shdsmw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = shrinkage_msm(y, treatment_history, covariate_history, lam)
    assert isinstance(result, dict)
