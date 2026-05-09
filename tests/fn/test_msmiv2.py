"""Tests for msmiv2.msm_iv."""
import numpy as np
import pytest
from moirais.fn.msmiv2 import msm_iv


def test_msmiv2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    instruments = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_iv(y, treatment_history, instruments, covariate_history)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmiv2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    instruments = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_iv(y, treatment_history, instruments, covariate_history)
    assert isinstance(result, dict)
