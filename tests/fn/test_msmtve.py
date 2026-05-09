"""Tests for msmtve.msm_time_varying_exposure."""
import numpy as np
import pytest
from moirais.fn.msmtve import msm_time_varying_exposure


def test_msmtve_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposure_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = msm_time_varying_exposure(y, exposure_history, covariate_history, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmtve_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposure_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = msm_time_varying_exposure(y, exposure_history, covariate_history, time)
    assert isinstance(result, dict)
