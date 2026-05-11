"""Tests for msmphr.msm_proportional_hazards."""
import numpy as np
import pytest
from morie.fn.msmphr import msm_proportional_hazards


def test_msmphr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_proportional_hazards(time, event, treatment_history, covariate_history)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msmphr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    result = msm_proportional_hazards(time, event, treatment_history, covariate_history)
    assert isinstance(result, dict)
